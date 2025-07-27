FROM nvcr.io/nvidia/nemo:24.07

COPY convert_mistral_2_hf.py /workspace/

COPY megatron_gpt_model.py /workspace/

RUN cp workspace/megatron_gpt_model.py /opt/NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py