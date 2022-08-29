mkdir logs_en
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py facebook/bart-large facebook/bart-large \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_bart_nodesc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda
                
# original botsdobem (desc_src - no_desc_trg)
python3 main.py facebook/bart-large facebook/bart-large \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_bart_descsrc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src

# original botsdobem (no_desc_src - desc_trg)
python3 main.py facebook/bart-large facebook/bart-large \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_bart_desctrg \
                english \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_trg 
                
# original botsdobem (desc_src - desc_trg)
python3 main.py facebook/bart-large facebook/bart-large \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_bart_desc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src \
                --describe_number_trg

# MT5
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py t5-base t5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_t5_nodesc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda
                
# original botsdobem (desc_src - no_desc_trg)
python3 main.py t5-base t5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_t5_descsrc \
                english \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda
                
# original botsdobem (no_desc_src - desc_trg)
python3 main.py t5-base t5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_t5_desctrg \
                english \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# original botsdobem (desc_src - desc_trg)
python3 main.py t5-base t5-base \
                'botsdobem synthetic' \
                30 \
                1e-4 \
                1 \
                5 \
                300 \
                logs_en/botsdobem_t5_desc \
                english \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# GPT-2
# original botsdobem (no_desc_src - no_desc_trg)
python3 main.py gpt2 gpt2 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs_en/botsdobem_gpt2_nodesc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda
                
# original botsdobem (desc_src - no_desc_trg)
python3 main.py gpt2 gpt2 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs_en/botsdobem_gpt2_descsrc \
                english \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda

# original botsdobem (no_desc_src - desc_trg)
python3 main.py gpt2 gpt2 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs_en/botsdobem_gpt2_desctrg \
                english \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# original botsdobem (desc_src - desc_trg)
python3 main.py gpt2 gpt2 \
                'botsdobem synthetic' \
                30 \
                1e-5 \
                2 \
                5 \
                300 \
                logs_en/botsdobem_gpt2_desc \
                english \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda