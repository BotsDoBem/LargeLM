mkdir logs
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py facebook/mbart-large-50 facebook/mbart-large-50 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs/botsdobem_mbart_nodesc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 
                
# original botsdobem (desc_src - no_desc_trg)
python3 main.py facebook/mbart-large-50 facebook/mbart-large-50 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs/botsdobem_mbart_descsrc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src \
                --src_lang pt_XX \
                --trg_lang pt_XX 

# original botsdobem (no_desc_src - desc_trg)
python3 main.py facebook/mbart-large-50 facebook/mbart-large-50 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs/botsdobem_mbart_desctrg \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_trg \
                --src_lang pt_XX \
                --trg_lang pt_XX 
                
# original botsdobem (desc_src - desc_trg)
python3 main.py facebook/mbart-large-50 facebook/mbart-large-50 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs/botsdobem_mbart_desc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src \
                --describe_number_trg \
                --src_lang pt_XX \
                --trg_lang pt_XX 

# MT5
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py google/mt5-base google/mt5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs/botsdobem_mt5_nodesc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda
                
# original botsdobem (desc_src - no_desc_trg)
python3 main.py google/mt5-base google/mt5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs/botsdobem_mt5_descsrc \
                portuguese \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda
                
# original botsdobem (no_desc_src - desc_trg)
python3 main.py google/mt5-base google/mt5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs/botsdobem_mt5_desctrg \
                portuguese \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# original botsdobem (desc_src - desc_trg)
python3 main.py google/mt5-base google/mt5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs/botsdobem_mt5_desc \
                portuguese \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# GPT-2
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py pierreguillou/gpt2-small-portuguese pierreguillou/gpt2-small-portuguese \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_gportuguese_nodesc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda
                
# original botsdobem (desc_src - no_desc_trg)
python3 main.py pierreguillou/gpt2-small-portuguese pierreguillou/gpt2-small-portuguese \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_gportuguese_descsrc \
                portuguese \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda

# original botsdobem (no_desc_src - desc_trg)
python3 main.py pierreguillou/gpt2-small-portuguese pierreguillou/gpt2-small-portuguese \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_gportuguese_desctrg \
                portuguese \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# original botsdobem (desc_src - desc_trg)
python3 main.py pierreguillou/gpt2-small-portuguese pierreguillou/gpt2-small-portuguese \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_gportuguese_desc \
                portuguese \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# BERT
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py neuralmind/bert-base-portuguese-cased neuralmind/bert-base-portuguese-cased \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_bertimbau \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

# original botsdobem (desc_src - no_desc_trg)
python3 main.py neuralmind/bert-base-portuguese-cased neuralmind/bert-base-portuguese-cased \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_bertimbau_descsrc \
                portuguese \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda

# original botsdobem (no_desc_src - desc_trg)
python3 main.py neuralmind/bert-base-portuguese-cased neuralmind/bert-base-portuguese-cased \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_bertimbau_desctrg \
                portuguese \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# original botsdobem (desc_src - desc_trg)
python3 main.py neuralmind/bert-base-portuguese-cased neuralmind/bert-base-portuguese-cased \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs/botsdobem_bertimbau_desc \
                portuguese \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda