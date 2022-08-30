mkdir results

# # bert - synthetic botsdobem
# python3 evaluate.py neuralmind/bert-base-portuguese-cased logs/botsdobem_synthetic_bertimbau/model \
#                 'botsdobem synthetic' \
#                 2 \
#                 300 \
#                 results/botsdobem_synthetic_bertimbau \
#                 portuguese \
#                 --verbose \
#                 --batch_status 16 \
#                 --cuda
                
# # bert - synthetic botsdobem (desc src)
# python3 evaluate.py neuralmind/bert-base-portuguese-cased logs/botsdobem_bertimbau_descsrc/model \
#                 'botsdobem synthetic' \
#                 2 \
#                 300 \
#                 results/botsdobem_bertimbau_descsrc \
#                 portuguese \
#                 --verbose \
#                 --describe_number_src \
#                 --batch_status 16 \
#                 --cuda
                
# # bert - synthetic botsdobem (desc trg)
# python3 evaluate.py neuralmind/bert-base-portuguese-cased logs/botsdobem_bertimbau_desctrg/model \
#                 'botsdobem synthetic' \
#                 2 \
#                 300 \
#                 results/botsdobem_bertimbau_descsrc \
#                 portuguese \
#                 --verbose \
#                 --describe_number_trg \
#                 --batch_status 16 \
#                 --cuda
                
# # bert - synthetic botsdobem (desc trg)
# python3 evaluate.py neuralmind/bert-base-portuguese-cased logs/botsdobem_bertimbau_desc/model \
#                 'botsdobem synthetic' \
#                 2 \
#                 300 \
#                 results/botsdobem_bertimbau_desc \
#                 portuguese \
#                 --verbose \
#                 --describe_number_src \
#                 --describe_number_trg \
#                 --batch_status 16 \
#                 --cuda

# mbart - synthetic botsdobem
python3 evaluate.py facebook/mbart-large-50 logs/botsdobem_mbart_nodesc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mbart_nodesc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --src_lang pt_XX \
                --trg_lang pt_XX 
                
# mbart - synthetic botsdobem (desc src)
python3 evaluate.py facebook/mbart-large-50 logs/botsdobem_mbart_descsrc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mbart_descsrc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src \
                --src_lang pt_XX \
                --trg_lang pt_XX 
                
# mbart - synthetic botsdobem (desc trg)
python3 evaluate.py facebook/mbart-large-50 logs/botsdobem_mbart_desctrg/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mbart_desctrg \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_trg \
                --src_lang pt_XX \
                --trg_lang pt_XX 

# mbart - synthetic botsdobem (desc src/trg)
python3 evaluate.py facebook/mbart-large-50 logs/botsdobem_mbart_desc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mbart_desc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda \
                --describe_number_src \
                --describe_number_trg \
                --src_lang pt_XX \
                --trg_lang pt_XX 

# MT5
# mt5 - synthetic botsdobem
python3 evaluate.py google/mt5-base logs/botsdobem_mt5_nodesc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mt5_nodesc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda
                
# mt5 - synthetic botsdobem (desc src)
python3 evaluate.py google/mt5-base logs/botsdobem_mt5_descsrc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mt5_descsrc \
                portuguese \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda
                
# mt5 - synthetic botsdobem (desc trg)
python3 evaluate.py google/mt5-base logs/botsdobem_mt5_desctrg/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mt5_desctrg \
                portuguese \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# mt5 - synthetic botsdobem (desc)
python3 evaluate.py google/mt5-base logs/botsdobem_mt5_desc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_mt5_desc \
                portuguese \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda

# gpt2 - synthetic botsdobem
python3 evaluate.py pierreguillou/gpt2-small-portuguese logs/botsdobem_gportuguese_nodesc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_gportuguese_nodesc \
                portuguese \
                --verbose \
                --batch_status 16 \
                --cuda

# gpt2 - synthetic botsdobem (desc src)
python3 evaluate.py pierreguillou/gpt2-small-portuguese logs/botsdobem_gportuguese_descsrc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_gportuguese_descsrc \
                portuguese \
                --verbose \
                --describe_number_src \
                --batch_status 16 \
                --cuda
                
# gpt2 - synthetic botsdobem (desc trg)
python3 evaluate.py pierreguillou/gpt2-small-portuguese logs/botsdobem_gportuguese_desctrg/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_gportuguese_desctrg \
                portuguese \
                --verbose \
                --describe_number_trg \
                --batch_status 16 \
                --cuda
                
# gpt2 - synthetic botsdobem (desc)
python3 evaluate.py pierreguillou/gpt2-small-portuguese logs/botsdobem_gportuguese_desc/model \
                'botsdobem synthetic' \
                2 \
                300 \
                results/botsdobem_gportuguese_desc \
                portuguese \
                --verbose \
                --describe_number_src \
                --describe_number_trg \
                --batch_status 16 \
                --cuda