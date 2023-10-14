from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, embed_dim = query.size()

        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(2, 3))  # dots scores

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_prob = F.softmax(attn_scores, dim=-1)

        attn_vec = torch.matmul(attn_prob, value)  # (batch_size, num_heads, seq_len, head_dim)
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)
        output = self.out_linear(attn_vec)
        return output


@dataclass
class SingleLossOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HaformerForFinetune(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # Initialize weights and apply final processing
        self.post_init()
        self.pooled = nn.AdaptiveMaxPool1d(1)
        self.attention = MultiHeadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads)

    def forward(
            self,
            input_ids_asm: Optional[torch.Tensor] = None,
            attention_mask_asm: Optional[torch.Tensor] = None,
            token_type_ids_asm: Optional[torch.Tensor] = None,
            position_ids_asm: Optional[torch.Tensor] = None,
            input_ids_hex: Optional[torch.Tensor] = None,
            attention_mask_hex: Optional[torch.Tensor] = None,
            token_type_ids_hex: Optional[torch.Tensor] = None,
            position_ids_hex: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SingleLossOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size_asm = input_ids_asm.size(0)
        num_sent_asm = input_ids_asm.size(1)
        input_ids_asm = input_ids_asm.view((-1, input_ids_asm.size(-1)))  # (bs * num_sent, len)
        attention_mask_asm = attention_mask_asm.view((-1, attention_mask_asm.size(-1)))  # (bs * num_sent len)
        if token_type_ids_asm is not None:
            token_type_ids_asm = token_type_ids_asm.view((-1, token_type_ids_asm.size(-1)))  # (bs * num_sent, len)

        input_ids_hex = input_ids_hex.view((-1, input_ids_hex.size(-1)))  # (bs * num_sent, len)
        attention_mask_hex = attention_mask_hex.view((-1, attention_mask_hex.size(-1)))  # (bs * num_sent len)
        if token_type_ids_hex is not None:
            token_type_ids_hex = token_type_ids_hex.view((-1, token_type_ids_hex.size(-1)))  # (bs * num_sent, len)

        outputs_asm = self.bert(
            input_ids_asm,
            attention_mask=attention_mask_asm,
            token_type_ids=token_type_ids_asm,
            position_ids=position_ids_asm,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_hex = self.bert(
            input_ids_hex,
            attention_mask=attention_mask_hex,
            token_type_ids=token_type_ids_hex,
            position_ids=position_ids_hex,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embedding_asm = outputs_asm.last_hidden_state
        embedding_hex = outputs_hex.last_hidden_state
        fuse_vector = self.fuse_asm_hex_vectors(embedding_asm, embedding_hex)

        fuse_output = fuse_vector.view(
            (batch_size_asm, num_sent_asm, fuse_vector.size(-1)))  # (bs, num_sent, hidden)
        z1, z2 = fuse_output[:, 0], fuse_output[:, 1]
        sim = Similarity(temp=self.config.temperature)
        z1_z2_cos = sim(z1.unsqueeze(1), z2.unsqueeze(0))

        labels = torch.arange(z1_z2_cos.size(0)).long().to(input_ids_asm.device)
        loss_fct = CrossEntropyLoss()
        contrastive_loss = loss_fct(z1_z2_cos, labels)

        if not return_dict:
            output = (fuse_output,) + outputs_asm[2:]
            return ((contrastive_loss,) + output) if contrastive_loss is not None else output

        return SingleLossOutput(
            loss=contrastive_loss,
            hidden_states=outputs_asm.hidden_states,
            attentions=outputs_asm.attentions,
        )

    def fuse_asm_hex_vectors(self, embedding_asm, embedding_hex):
        fuse_asm_hex_vector = self.attention(embedding_asm, embedding_hex, embedding_asm)
        fuse_asm_hex_vector = fuse_asm_hex_vector.permute(0, 2, 1)
        pooled_fuse_vec = self.pooled(fuse_asm_hex_vector)
        pooled_fuse_vec = pooled_fuse_vec.squeeze(-1)
        return pooled_fuse_vec
