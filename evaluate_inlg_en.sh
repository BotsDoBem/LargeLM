mkdir results_en

# bart - synthetic botsdobem
python3 evaluate.py facebook/bart-large logs_en/botsdobem_bart_nodesc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_bart_nodesc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda
                
# bart - synthetic botsdobem (desc src)
python3 evaluate.py facebook/bart-large logs_en/botsdobem_bart_descsrc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_bart_descsrc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src
                
# bart - synthetic botsdobem (desc trg)
python3 evaluate.py facebook/bart-large logs_en/botsdobem_bart_desctrg/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_bart_desctrg \
                english \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_trg

# bart - synthetic botsdobem (desc src/trg)
python3 evaluate.py facebook/bart-large logs_en/botsdobem_bart_desc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_bart_desc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src \
                --describe_number_trg

# T5
# t5 - synthetic botsdobem
python3 evaluate.py t5-base logs_en/botsdobem_t5_nodesc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_t5_nodesc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda
                
# t5 - synthetic botsdobem (desc src)
python3 evaluate.py t5-base logs_en/botsdobem_t5_descsrc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_t5_descsrc \
                english \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda
                
# t5 - synthetic botsdobem (desc trg)
python3 evaluate.py t5-base logs_en/botsdobem_t5_desctrg/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_t5_desctrg \
                english \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# t5 - synthetic botsdobem (desc)
python3 evaluate.py t5-base logs_en/botsdobem_t5_desc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_t5_desc \
                english \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# gpt2 - synthetic botsdobem
python3 evaluate.py gpt2 logs_en/botsdobem_gpt2_nodesc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_gpt2_nodesc \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

# gpt2 - synthetic botsdobem (desc src)
python3 evaluate.py gpt2 logs_en/botsdobem_gpt2_descsrc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_gpt2_descsrc \
                english \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda
                
# gpt2 - synthetic botsdobem (desc trg)
python3 evaluate.py gpt2 logs_en/botsdobem_gpt2_desctrg/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_gpt2_desctrg \
                english \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda
                
# gpt2 - synthetic botsdobem (desc)
python3 evaluate.py gpt2 logs_en/botsdobem_gpt2_desc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results_en/botsdobem_gpt2_desc \
                english \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda