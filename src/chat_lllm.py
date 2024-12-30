import torch

from decoder_only_model import DecoderOnlyModel


class ChatLLM:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.model = DecoderOnlyModel()

    def chat(self, prompt):
        input_ids = tokenizer.encode(prompt)
        generated_ids = input_ids

        for _ in range(self.max_seq_len - len(input_ids[0])):
            logits = self.model
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
            
            if next_token_id.item() == "endsid":
                break
        return tokenizer.decode(generated_ids)