import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler


class TransformerEmbedding(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 n_positions: int,
                 n_segments: int = 3,
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        super().__init__()

        self.scaling = embed_dim ** 0.5

        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=embed_dim,
                                             padding_idx=padding_idx)

        self.segment_embedding = nn.Embedding(num_embeddings=n_segments,
                                              embedding_dim=embed_dim,
                                              padding_idx=padding_idx)

        self.positional_embedding = nn.Embedding(num_embeddings=n_positions,
                                                 embedding_dim=embed_dim,
                                                 padding_idx=padding_idx)

        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                token_sequence: torch.tensor,
                segment_indices: torch.tensor,
                position_indices: torch.tensor) -> torch.tensor:
        """
        :param token_sequence: [sequence_length, batch_size]
        :param segment_indices: [sequence_length, batch_size]
        :param position_indices: [sequence_length, batch_size]
        :return: [sequence_length, batch_size, embed_dim]
        """

        token_emb = self.token_embeddings(token_sequence) * self.scaling
        segment_emb = self.segment_embedding(segment_indices)
        position_emb = self.positional_embedding(position_indices)

        emb = token_emb + segment_emb + position_emb
        emb = self.dropout(self.layer_norm(emb))

        return emb


class MultiHeadSelfAttention(nn.Module):
    """
    Need re-implement this layer for custom attention mask with shape: [batch_size, sequence_len, sequence_len]
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim

        self.scaling = self.head_dim ** -0.5

        self.in_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self,
                    embed: torch.tensor,
                    batch_size: int,
                    sequence_len: int) -> torch.tensor:
        """
        From [batch_size * self.num_heads, sequence_len, sequence_len]
        To [batch_size, self.num_heads, sequence_len, sequence_len]
        """
        return embed.view(batch_size, self.num_heads, sequence_len, sequence_len)

    def join_heads(self,
                   embed: torch.tensor,
                   batch_size: int,
                   sequence_len: int) -> torch.tensor:
        """
        From [batch_size, self.num_heads, sequence_len, sequence_len]
        To [batch_size * self.num_heads, sequence_len, sequence_len]
        """
        return embed.view(batch_size * self.num_heads, sequence_len, sequence_len)

    def forward(self,
                query: torch.tensor,
                padding_mask: Optional[torch.tensor] = None,
                attention_mask: Optional[torch.tensor] = None,
                need_weights: bool = False) -> torch.tensor:
        """
        :param query: [sequence_length, batch_size, embed_dim]
        :param padding_mask: [batch_size, sequence_len]
        :param attention_mask: [batch_size, sequence_len, sequence_len]
        :param need_weights: bool
        :return: [sequence_length, batch_size, embed_dim]
        """

        sequence_len, batch_size, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        query, key, value = self.in_projection(query).chunk(3, dim=-1)

        query *= self.scaling

        # [batch_size * self.num_heads, sequence_len, self.head_dim]
        query = query.contiguous().view(sequence_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(sequence_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(sequence_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # [batch_size * self.num_heads, sequence_len, sequence_len]
        attention_scores = torch.bmm(query, key.transpose(1, 2))

        # [batch_size, self.num_heads, sequence_len, sequence_len]
        attention_scores = self.split_heads(attention_scores, batch_size, sequence_len)

        # fp16 compatibility
        parameters_type = next(self.parameters()).dtype

        if attention_mask is not None:
            assert attention_mask.size(1) == sequence_len
            assert attention_mask.size(2) == sequence_len
            # [batch_size, 1, sequence_len, sequence_len]
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=parameters_type)
            attention_scores += attention_mask

        if padding_mask is not None:
            assert padding_mask.size(0) == batch_size
            assert padding_mask.size(1) == sequence_len
            # padding_mask = [batch_size, sequence_len]
            attention_scores = attention_scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2),
                float(-10000.),
            )

        # [batch_size * self.num_heads, sequence_len, sequence_len]
        attention_scores = self.join_heads(attention_scores, batch_size, sequence_len)

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        # [batch_size * self.num_heads, sequence_len, sequence_len]
        attention_scores = F.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout(attention_scores)

        # attention_scores = [batch_size * self.num_heads, sequence_len, sequence_len]
        # value = [batch_size * self.num_heads, sequence_len, self.head_dim]
        # [batch_size * self.num_heads, sequence_len, self.head_dim]
        attention_output = torch.bmm(attention_scores, value)

        # [sequence_len, batch_size, embed_dim]
        attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_len, batch_size, embed_dim)
        attention_output = self.out_projection(attention_output)

        # for visualize attention scores
        if need_weights:
            # [batch_size, self.num_heads, sequence_len, sequence_len]
            attention_scores = self.split_heads(attention_scores, batch_size, sequence_len)
        else:
            attention_scores = None

        return attention_output, attention_scores


class PositionWiseFeedForwardLayer(nn.Module):
    """
    Conv1d idea from GPT-2
    It's the same operation as linear projection,
    but better because cnn more efficient for parallel computing across sequence
    """

    def __init__(self, embed_dim: int, increased_dim: int, dropout: float = 0.1):
        super().__init__()

        self.increase = nn.Conv1d(in_channels=embed_dim, out_channels=increased_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.decrease = nn.Conv1d(in_channels=increased_dim, out_channels=embed_dim, kernel_size=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: [*, *, embed_dim]
        :return: [*, *, embed_dim]
        """

        x = torch.relu(self.increase(x))
        x = self.dropout(x)
        x = self.decrease(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, model_dim: int, num_heads: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(embed_dim=model_dim,
                                                     num_heads=num_heads,
                                                     dropout=dropout)

        self.position_wise_feed_forward = PositionWiseFeedForwardLayer(embed_dim=model_dim,
                                                                       increased_dim=feed_forward_dim,
                                                                       dropout=dropout)

        self.norm_attention = nn.LayerNorm(model_dim)
        self.norm_feed_forward = nn.LayerNorm(model_dim)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_feed_forward = nn.Dropout(dropout)

    def forward(self,
                x: torch.tensor,
                attention_mask: Optional[torch.tensor] = None,
                padding_mask: Optional[torch.tensor] = None) -> torch.tensor:
        """
        :param x: [sequence_length, batch_size, embed_dim]
        :param padding_mask: [batch_size, sequence_len]
        :param attention_mask: [batch_size, sequence_len, sequence_len]
        :return: [sequence_length, batch_size, embed_dim]
        """

        hidden = self.self_attention(x, padding_mask=padding_mask, attention_mask=attention_mask)[0]
        x = x + self.dropout_attention(hidden)
        x = self.norm_attention(x)

        # [sequence_length, batch_size, embed_dim] -> [batch_size, embed_dim, sequence_length]
        x = x.permute(1, 2, 0)
        hidden = self.position_wise_feed_forward(x)
        x = x + self.dropout_feed_forward(hidden)

        # [batch_size, embed_dim, sequence_length] -> [sequence_length, batch_size, embed_dim]
        x = x.permute(2, 0, 1)
        x = self.norm_feed_forward(x)

        return x


class UnifiedTransformer(nn.Module):
    """
    Unidirectional decoding conditioned on bidirectional encoding
    This setup have more attention interactions over words in query (and context)
    and response (for response generation) than traditional transformer, but have less training signal from batch
    paper: https://arxiv.org/abs/1905.03197
    """

    def __init__(self,
                 model_dim: int = 768,
                 num_heads: int = 12,
                 feed_forward_dim: int = 3072,
                 num_layers: int = 12,
                 response_segment_index: int = 1,
                 query_segment_index: int = 2,
                 context_segment_index: Optional[int] = 3,
                 weight_tying: bool = False,
                 vocab_size: int = 32000,
                 n_positions: int = 65,
                 dropout: float = 0.1,
                 padding_idx: int = 0,
                 initializer_range: float = 0.02,
                 masking_type: str = 'seq2seq'):
        super().__init__()

        self.initializer_range = initializer_range
        self.padding_idx = padding_idx
        self.masking_type = masking_type
        assert self.masking_type in ['causal', 'lm', 'seq2seq'], 'Available masking_types: causal (lm) and seq2seq'

        self.response_segment_index = response_segment_index
        self.query_segment_index = query_segment_index
        self.context_segment_index = context_segment_index

        n_segments = 4 if self.context_segment_index is not None else 3

        self.embedding = TransformerEmbedding(embed_dim=model_dim,
                                              vocab_size=vocab_size,
                                              n_positions=n_positions,
                                              n_segments=n_segments,
                                              dropout=dropout,
                                              padding_idx=padding_idx)

        self.layers = nn.ModuleList([
            TransformerEncoder(
                model_dim=model_dim,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(model_dim, vocab_size, bias=False)

        self.apply(self.init_weights)

        if weight_tying:
            self.head.weight = self.embedding.token_embeddings.weight

    def init_weights(self, module: nn.Module):
        """
        From GPT-2
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def generate_square_subsequent_mask(sequence_len: int, fill_value: float = -10000.) -> torch.tensor:
        mask = (torch.triu(torch.ones(sequence_len, sequence_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, fill_value).masked_fill(mask == 1, float(0.0))

        return mask

    @staticmethod
    def generate_seq2seq_mask(sequence_len: int,
                              non_masked_lengths: torch.tensor,
                              fill_value: float = -10000.) -> torch.tensor:
        mask = (torch.triu(torch.ones(sequence_len, sequence_len)) == 1).transpose(0, 1)
        mask = mask.unsqueeze(0).repeat(non_masked_lengths.size(0), 1, 1)

        for i in range(non_masked_lengths.size(0)):
            mask[i, :non_masked_lengths[i], :non_masked_lengths[i]] = True

        mask = mask.float().masked_fill(mask == 0, fill_value).masked_fill(mask == 1, float(0.0))

        return mask

    def set_causal(self):
        self.masking_type = 'causal'

    def set_seq2seq(self):
        self.masking_type = 'seq2seq'

    def forward(self,
                source_sequence: torch.tensor,
                segment_indices: torch.tensor,
                position_indices: torch.tensor,
                targets: Optional[torch.tensor] = None,
                attention_mask: Optional[torch.tensor] = None,
                padding_mask: Optional[torch.tensor] = None) -> torch.tensor:
        """
        :param source_sequence: [batch_size, sequence_len]
        :param segment_indices: [sequence_length, batch_size]
        :param position_indices: [sequence_length, batch_size]
        :param targets: [batch_size, sequence_len]
        :param attention_mask: [batch_size, sequence_len, sequence_len]
        :param padding_mask: [batch_size, sequence_len]
        :return: [batch_size, sequence_len]
        """
        if attention_mask is None:
            if self.masking_type in ['causal', 'lm']:
                attention_mask = self.generate_square_subsequent_mask(sequence_len=source_sequence.size(-1))
                # for the same shape len as seq2seq mask
                attention_mask = attention_mask.unsqueeze(0)
            elif self.masking_type in ['seq2seq'] or not self.training:
                non_masked_lengths = (segment_indices == self.query_segment_index) \
                                     | (segment_indices == self.context_segment_index)
                non_masked_lengths = non_masked_lengths.sum(-1)

                attention_mask = self.generate_seq2seq_mask(sequence_len=source_sequence.size(-1),
                                                            non_masked_lengths=non_masked_lengths)

            attention_mask = attention_mask.to(source_sequence.device)

        if padding_mask is None:
            padding_mask = source_sequence == self.padding_idx
            padding_mask = padding_mask.to(source_sequence.device)

        hidden = self.embedding(token_sequence=source_sequence,
                                segment_indices=segment_indices,
                                position_indices=position_indices)

        # [batch_size, sequence_len] -> [sequence_len, batch_size]
        hidden = hidden.transpose(0, 1)

        for layer in self.layers:
            hidden = layer(x=hidden, attention_mask=attention_mask, padding_mask=padding_mask)

        # [sequence_len, batch_size, model_dim] -> [batch_size, sequence_len, model_dim]
        logits = hidden.transpose(0, 1)

        if targets is not None and self.training:
            response_mask = segment_indices == self.response_segment_index
            logits = logits[(targets != self.padding_idx) & response_mask, :]
            targets = targets[(targets != self.padding_idx) & response_mask]

        logits = self.head(logits)

        return logits, targets


class GPT(nn.Module):
    """
    Just GPT
    """

    def __init__(self,
                 model_dim: int = 768,
                 num_heads: int = 12,
                 feed_forward_dim: int = 3072,
                 num_layers: int = 12,
                 query_segment_index: int = 2,
                 context_segment_index: Optional[int] = 3,
                 weight_tying: bool = False,
                 vocab_size: int = 32000,
                 n_positions: int = 64,
                 dropout: float = 0.1,
                 padding_idx: int = 0,
                 initializer_range: int = 0.02):
        super().__init__()

        self.initializer_range = initializer_range
        self.padding_idx = padding_idx

        self.query_segment_index = query_segment_index
        self.context_segment_index = context_segment_index

        n_segments = 4 if self.context_segment_index is not None else 3

        self.embedding = TransformerEmbedding(embed_dim=model_dim,
                                              vocab_size=vocab_size,
                                              n_positions=n_positions,
                                              n_segments=n_segments,
                                              dropout=dropout,
                                              padding_idx=padding_idx)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=model_dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=feed_forward_dim,
                                                     dropout=dropout),
            num_layers=num_layers
        )

        self.head = nn.Linear(model_dim, vocab_size, bias=False)

        self.apply(self.init_weights)

        if weight_tying:
            self.head.weight = self.embedding.token_embeddings.weight

    def init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def generate_square_subsequent_mask(sequence_len: int) -> torch.tensor:
        mask = (torch.triu(torch.ones(sequence_len, sequence_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self,
                source_sequence: torch.tensor,
                segment_indices: torch.tensor,
                position_indices: torch.tensor,
                attention_mask: Optional[torch.tensor] = None,
                padding_mask: Optional[torch.tensor] = None) -> torch.tensor:
        """
        :param source_sequence: [batch_size, sequence_len]
        :param segment_indices: [sequence_length, batch_size]
        :param position_indices: [sequence_length, batch_size]
        :param attention_mask: [sequence_len, sequence_len]
        :param padding_mask: [batch_size, sequence_len]
        :return: [batch_size, sequence_len]
        """
        if attention_mask is None:
            attention_mask = self.generate_square_subsequent_mask(sequence_len=source_sequence.size(1))
            attention_mask = attention_mask.to(source_sequence.device)

        if padding_mask is None:
            padding_mask = (source_sequence == self.padding_idx).to(source_sequence.device)

        hidden = self.embedding(source_sequence,
                                segment_indices=segment_indices,
                                position_indices=position_indices)

        hidden = hidden.transpose(0, 1)

        hidden = self.transformer(hidden, attention_mask, padding_mask)

        logits = self.head(hidden).transpose(0, 1)

        return logits


class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing: float = 0.1, use_kl: bool = False, ignore_index: int = -100):
        super().__init__()

        assert 0 <= smoothing < 1

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.use_kl = use_kl

    def smooth_one_hot(self, true_labels: torch.Tensor, classes: int) -> torch.Tensor:

        confidence = 1.0 - self.smoothing

        with torch.no_grad():
            true_dist = torch.empty(size=(true_labels.size(0), classes), device=true_labels.device)
            true_dist.fill_(self.smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

        return true_dist

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param prediction: [batch_size, num_classes]
        :param target: [batch_size]
        :param mask: [batch_size, num_classes] True if need
        :return: scalar
        """

        if mask is not None:
            prediction = prediction[(target != self.ignore_index) & mask, :]
            target = target[(target != self.ignore_index) & mask]
        else:
            prediction = prediction[target != self.ignore_index, :]
            target = target[target != self.ignore_index]

        prediction = torch.nn.functional.log_softmax(prediction, dim=1)

        target_smoothed_dist = self.smooth_one_hot(target, classes=prediction.size(-1))

        if self.use_kl:
            loss = F.kl_div(prediction, target_smoothed_dist, reduction='batchmean')
        else:
            loss = torch.mean(torch.sum(-target_smoothed_dist * prediction, dim=-1))

        return loss


class NoamScheduler(_LRScheduler):
    """
    From "Attention Is All You Need"
    paper: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, optimizer, model_dim, warmup_steps=4000):
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.model_dim ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [lr * scale for lr in self.base_lrs]
