python train.py \
--config=config/paa_transformer/paa_transformer-small.yml \
--dataset=convai2 \
--lr=1e-6 \
--gated=yes \
--fusion_mode=pr-cr \
--auto_tau=accurate \
--auto_tau_numerator=persona \
--response_gated=no \
--shared_enc=no \
--shared_crossattention=no \
--add_persona_to_decoder=yes \
--add_persona_indicator=yes \
--add_role_indicator=yes