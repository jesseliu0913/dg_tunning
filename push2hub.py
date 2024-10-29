from huggingface_hub import Repository


repo = Repository(local_dir="oneround_meditron_7b", clone_from="JesseLiu/oneround_meditron_7b")
repo.push_to_hub()

