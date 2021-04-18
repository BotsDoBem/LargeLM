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
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 

python3 main.py google/mt5-large \
                'botsdobem original' \
                30 \
                1e-3 \
                2 \
                2 \
                5 \
                256 \
                logs/botsdobem_mt5 \
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
                --verbose \
                --batch_status 16 \
                --cuda

# webnlg
python3 main.py bert-large-cased \
                'webnlg' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/webnlg_bert \
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
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py google/t5-large \
                'webnlg' \
                30 \
                1e-4 \
                4 \
                4 \
                5 \
                512 \
                logs/webnlg_t5 \
                --verbose \
                --batch_status 16 \
                --cuda

# e2e
python3 main.py bert-large-cased \
                'e2e' \
                30 \
                1e-5 \
                4 \
                4 \
                5 \
                512 \
                logs/e2e_bert \
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
                --verbose \
                --batch_status 16 \
                --cuda

python3 main.py google/t5-large \
                'e2e' \
                30 \
                1e-4 \
                4 \
                4 \
                5 \
                512 \
                logs/e2e_t5 \
                --verbose \
                --batch_status 16 \
                --cuda