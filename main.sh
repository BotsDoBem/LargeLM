mkdir logs
python3 main.py neuralmind/bert-base-portuguese-cased \
                'botsdobem original' \
                30 \
                1e-5 \
                2 \
                2 \
                5 \
                256 \
                logs/botsdobem_bertimbau \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py facebook/mbart-large-50 \
                'botsdobem original' \
                30 \
                1e-5 \
                2 \
                2 \
                5 \
                256 \
                logs/botsdobem_mbart \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 

python3 main.py google/mt5-large \
                'botsdobem original' \
                30 \
                1e-3 \
                1 \
                1 \
                5 \
                256 \
                logs/botsdobem_mt5 \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py pierreguillou/gpt2-small-portuguese \
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
python3 main.py bert-base-cased \
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

python3 main.py facebook/bart-large \
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

python3 main.py google/t5-large \
                'webnlg' \
                30 \
                1e-4 \
                1 \
                1 \
                5 \
                512 \
                logs/webnlg_t5 \
                english \
                --verbose \
                --batch_status 16 \
                --cuda

# e2e
python3 main.py bert-base-cased \
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

python3 main.py facebook/bart-large \
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

python3 main.py google/t5-large \
                'e2e' \
                30 \
                1e-4 \
                1 \
                1 \
                5 \
                512 \
                logs/e2e_t5 \
                english \
                --verbose \
                --batch_status 16 \
                --cuda