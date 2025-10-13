
# Run the experiments
run-mem0-add:
	python run_experiments.py --technique_type mem0 --method add

run-mem0-search:
	python run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30

run-mem0-plus-add:
	python run_experiments.py --technique_type mem0 --method add --is_graph

run-mem0-plus-search:
	python run_experiments.py --technique_type mem0 --method search --is_graph --output_folder results/ --top_k 30

run-rag:
	python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	python run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	python run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	python run_experiments.py --technique_type zep --method add --output_folder results/

run-zep-search:
	python run_experiments.py --technique_type zep --method search --output_folder results/

run-openai:
	python run_experiments.py --technique_type openai --output_folder results/

run-personamem-add:
        python run_experiments_personamem.py --method add --questions-file dataset/PersonaMem/questions_32k.csv --contexts-file dataset/PersonaMem/shared_contexts_32k.jsonl

run-personamem-dry-run:
        python run_experiments_personamem.py --method add --questions-file dataset/PersonaMem/questions_32k.csv --contexts-file dataset/PersonaMem/shared_contexts_32k.jsonl --dry-run

run-personamem-search:
        python run_experiments_personamem.py --method search --questions-file dataset/PersonaMem/questions_32k.csv --contexts-file dataset/PersonaMem/shared_contexts_32k.jsonl --output-folder results --top-k 30
