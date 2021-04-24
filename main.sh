mkdir logs
python3 main.py neuralmind/bert-base-portuguese-cased neuralmind/bert-base-portuguese-cased \
                'botsdobem original' \
                30 \
                1e-5 \
                2 \
                2 \
                5 \
                300 \
                logs/botsdobem_bertimbau \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py facebook/mbart-large-50 facebook/mbart-large-50 \
                'botsdobem original' \
                30 \
                1e-5 \
                2 \
                2 \
                5 \
                300 \
                logs/botsdobem_mbart \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 

python3 main.py google/mt5-base google/mt5-base \
                'botsdobem original' \
                30 \
                1e-3 \
                2 \
                2 \
                5 \
                300 \
                logs/botsdobem_mt5 \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py pierreguillou/gpt2-small-portuguese pierreguillou/gpt2-small-portuguese \
                'botsdobem original' \
                30 \
                1e-5 \
                2 \
                2 \
                5 \
                300 \
                logs/botsdobem_gportuguese \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

# webnlg
python3 main.py bert-base-cased bert-base-cased \
                'webnlg' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/webnlg_bert \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py facebook/bart-large facebook/bart-large \
                'webnlg' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/webnlg_bart \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py t5-base t5-base \
                'webnlg' \
                30 \
                1e-4 \
                4 \
                4 \
                5 \
                512 \
                logs/webnlg_t5 \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py gpt2 gpt2 \
                'webnlg' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/webnlg_gpt \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

# e2e
python3 main.py bert-base-cased bert-base-cased \
                'e2e' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/e2e_bert \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py facebook/bart-large facebook/bart-large \
                'e2e' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/e2e_bart \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py t5-base t5-base \
                'e2e' \
                30 \
                1e-4 \
                4 \
                4 \
                5 \
                512 \
                logs/e2e_t5 \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py gpt2 gpt2 \
                'e2e' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/e2e_gpt \
                english \
                --verbose \
                --batch_status 16 \
                --cuda