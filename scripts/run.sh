#Caltech
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/caltech101_names_addattris.json --image-root /data/Your_dataset_path/Caltech101/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/caltech101_names_addattris.json --image-root /data/Your_dataset_path/Caltech101/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Pets
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/pets_names_addattris2.json --image-root /data/Your_dataset_path/OxfordPets/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.25 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/pets_names_addattris2.json --image-root /data/Your_dataset_path/OxfordPets/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Cars
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/cars_names_addattris_yearhead.json --image-root /data/Your_dataset_path/StanfordCars/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.25 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/cars_names_addattris_yearhead.json --image-root /data/Your_dataset_path/StanfordCars/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Flowers
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/flower102_names_addattris.json --image-root /data/Your_dataset_path/flower/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/flower102_names_addattris.json --image-root /data/Your_dataset_path/flower/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#DTD
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/dtd_names_addattris.json --image-root /data/Your_dataset_path/DescribableTextures/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/dtd_names_addattris.json --image-root /data/Your_dataset_path/DescribableTextures/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#UCF101
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/UCF101_names_addattris_new.json --image-root /data/Your_dataset_path/UCF101/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/UCF101_names_addattris_new.json --image-root /data/Your_dataset_path/UCF101/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#EuroSAT
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/EuroSATtest_names_addattris.json --image-root /data/Your_dataset_path/EuroSAT/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/EuroSATtest_names_addattris.json --image-root /data/Your_dataset_path/EuroSAT/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Food101
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/food101_names_addattris.json --image-root /data/Your_dataset_path/Food101/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/food101_names_addattris.json --image-root /data/Your_dataset_path/Food101/test --entropy-threshold-class 99  --consolidate-every 160 --novelty-theta 0.3  --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#SUN397
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/Sun_names_addattris.json --image-root /data/Your_dataset_path/SUN397/test1 --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/Sun_names_addattris.json --image-root /data/Your_dataset_path/SUN397/test1 --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Aircraft
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path /home/Your_baseknowledge_path/aircraft_names_addattris.json --image-root /data/Your_dataset_path/aircraft/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/aircraft_names_addattris.json --image-root /data/Your_dataset_path/aircraft/test --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Counter-animal-easy
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/animal_atri_foreasy.json --image-root /data/Your_dataset_path/LAION-final_easy --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/animal_atri_foreasy.json --image-root /data/Your_dataset_path/LAION-final_easy --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#Counter-animal-hard
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/animal_atri_forhard.json --image-root /data/Your_dataset_path/LAION-final_hard --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/animal_atri_forhard.json --image-root /data/Your_dataset_path/LAION-final_hard --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#imagenet-A
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/imagenet-a-addattris.json --image-root /data/Your_dataset_path/imagenet-a --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/imagenet-a-addattris.json --image-root /data/Your_dataset_path/imagenet-a --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#imagenet-V2
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/imagenet-v-addattris.json --image-root /data/Your_dataset_path/imagene-v/imagenetv2-matched-frequency-format-val --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/imagenet-v-addattris.json --image-root /data/Your_dataset_path/imagene-v/imagenetv2-matched-frequency-format-val --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#imagenet-R
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/imagenet-r-classnames-addattris.json --image-root /data/Your_dataset_path/imagenet-r --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/imagenet-r-classnames-addattris.json --image-root /data/Your_dataset_path/imagenet-r --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"


#imagenet-S
#ViT-B/16
python run_graph_eval.py --model-path /data/Your_model_path/ViT-B-16.pt --json-path  /home/Your_baseknowledge_path/imagenet-S-classnames-addattris.json --image-root /data/Your_dataset_path/imagenet-sketch/sketch --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"
#RN50
python run_graph_eval.py --model-path /data/Your_model_path/RN50.pt --json-path  /home/Your_baseknowledge_path/imagenet-S-classnames-addattris.json --image-root /data/Your_dataset_path/imagenet-sketch/sketch --entropy-threshold-class 99  --consolidate-every 160  --novelty-theta 0.3 --prune-every 240  --min-hits-to-keep 80 --llm-model gpt-3.5-turbo  --openai-api-key "xxxx"



