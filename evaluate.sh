mkdir results
# original botsdobem
python3 evaluate.py neuralmind/bert-base-portuguese-cased logs/botsdobem_bertimbau/model \
                'botsdobem original' \
                2 \
                300 \
                results/botsdobem_bertimbau \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py facebook/mbart-large-50 logs/botsdobem_mbart/model \
                'botsdobem original' \
                2 \
                300 \
                results/botsdobem_mbart \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 

python3 evaluate.py google/mt5-base logs/botsdobem_mt5/model \
                'botsdobem original' \
                2 \
                300 \
                results/botsdobem_mt5 \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py pierreguillou/gpt2-small-portuguese logs/botsdobem_gportuguese/model \
                'botsdobem original' \
                2 \
                300 \
                results/botsdobem_gportuguese \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

# synthetic botsdobem
python3 evaluate.py neuralmind/bert-base-portuguese-cased logs/botsdobem_synthetic_bertimbau/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_synthetic_bertimbau \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py facebook/mbart-large-50 logs/botsdobem_synthetic_mbart/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_synthetic_mbart \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 

python3 evaluate.py google/mt5-base logs/botsdobem_synthetic_mt5/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_synthetic_mt5 \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py pierreguillou/gpt2-small-portuguese logs/botsdobem_synthetic_gportuguese/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_synthetic_gportuguese \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

# webnlg
python3 evaluate.py bert-base-cased logs/webnlg_bert/model \
                'webnlg' \
                4 \
                512 \
                results/webnlg_bert \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py facebook/bart-large logs/webnlg_bart/model \
                'webnlg' \
                4 \
                512 \
                results/webnlg_bart \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py t5-base logs/webnlg_t5/model \
                'webnlg' \
                4 \
                512 \
                results/webnlg_t5 \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py gpt2 logs/webnlg_gpt/model \
                'webnlg' \
                4 \
                512 \
                results/webnlg_gpt \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

# e2e
python3 evaluate.py bert-base-cased logs/e2e_bert/model \
                'e2e' \
                4 \
                512 \
                results/e2e_bert \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py facebook/bart-large logs/e2e_bart/model \
                'e2e' \
                4 \
                512 \
                results/e2e_bart \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py t5-base logs/e2e_t5/model \
                'e2e' \
                4 \
                512 \
                results/e2e_t5 \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 evaluate.py gpt2 logs/e2e_gpt/model \
                'e2e' \
                4 \
                512 \
                results/e2e_gpt \
                english \
                --verbose \
                --batch_status 16 \
                --cuda