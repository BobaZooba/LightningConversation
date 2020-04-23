import torch
from torch.nn import functional as F
from src.tokenizer import Tokenizer
from src.data import BatchPreparing
from src.neuro import Seq2SeqLM
from typing import Optional, List


class Decoder:

    def __init__(self, tokenizer_path: str, model: Seq2SeqLM, max_turns: int = 64, device: Optional[str] = None):

        self.tokenizer = Tokenizer(tokenizer_path=tokenizer_path)

        self.batch_preparer = BatchPreparing(sep_index=self.tokenizer.sep_index,
                                             context_index=self.tokenizer.context_index,
                                             pad_index=self.tokenizer.pad_index)

        self.model = model

        self.max_turns = max_turns

        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

        self.model.eval()

    def tokenize(self, query: str, context: Optional[List[str]] = None) -> List[List[int]]:

        if context is not None and self.tokenizer.context_token is not None:
            text = self.tokenizer.context_token.join(context + [query]) + ' ' + self.tokenizer.sep_token
        else:
            text = query + ' ' + self.tokenizer.sep_token

        tokens = self.tokenizer([text])

        return tokens

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-10000.):
        assert logits.dim() == 1
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def predict(self, tokens: List[List[int]]):

        source_sequence, _, segment_indices, position_indices = self.batch_preparer.prepare_batch(tokens)

        source_sequence = source_sequence.to(self.device)
        segment_indices = segment_indices.to(self.device)
        position_indices = position_indices.to(self.device)

        with torch.no_grad():
            logits, _ = self.model.forward(source_sequence, segment_indices, position_indices)

        return logits

    def greedy_generate(self, query: str, context: Optional[List[str]] = None) -> str:

        answer_indices = list()

        tokens = self.tokenize(query, context)

        for _ in range(self.max_turns):

            logits = self.predict(tokens)

            predicted_index = logits[:, -1, :].argmax().detach().cpu().item()

            if predicted_index == self.tokenizer.eos_index:
                break

            answer_indices.append(predicted_index)
            tokens[0].insert(-1, predicted_index)

        response = self.tokenizer.bpe.decode(answer_indices)[0]

        return response

    def nucleous_generate(self, query: str, context: Optional[List[str]] = None,
                          temperature: float = 1., top_k: int = 0, top_p: float = 0.7) -> str:

        answer_indices = list()

        tokens = self.tokenize(query, context)

        for _ in range(self.max_turns):

            logits = self.predict(tokens)

            logits = logits[0, -1, :] / temperature
            filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)

            pred_index = next_token.detach().cpu().item()

            if pred_index == self.tokenizer.eos_index:
                break

            answer_indices.append(pred_index)
            tokens[0].insert(-1, pred_index)

        response = self.tokenizer.bpe.decode(answer_indices)[0]

        return response

    def beam_decoding(self, query: str, context: Optional[List[str]] = None, temperature: float = 1.):
        # answer_indices = list()
        #
        # tokens = self.tokenize(query, context)
        #
        # for _ in range(self.max_turns):
        #     logits = self.predict(tokens)
        #
        #     logits = logits[0, -1, :] / temperature

        raise NotImplementedError
