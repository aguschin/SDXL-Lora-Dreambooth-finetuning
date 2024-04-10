set -x
for dir in $(ls -d DreamboothSDXL* | tac); do
    if [ -d "$dir" ]; then
        python generate_finetuned.py "$dir"
    fi
done
