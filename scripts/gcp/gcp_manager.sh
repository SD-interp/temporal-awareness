#!/bin/bash

# GCP Instance Manager for Temporal Awareness Experiments

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# GCP Configuration (use env vars or defaults)
PROJECT="${GCP_PROJECT:-new-one-82feb}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-temporal-awareness-exp}"
BUCKET="${GCS_BUCKET:-gs://temporal-grounding-gpt2-82feb}"

function create_instance() {
    echo "Creating GCP instance..."
    gcloud compute instances create $INSTANCE \
        --project=$PROJECT \
        --zone=$ZONE \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --maintenance-policy=TERMINATE \
        --preemptible \
        --scopes=cloud-platform

    echo ""
    echo "Instance created!"
    echo "Wait 1-2 minutes for startup, then run: $0 ssh"
}

function ssh_instance() {
    gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE
}

function delete_instance() {
    echo "Deleting instance..."
    gcloud compute instances delete $INSTANCE --project=$PROJECT --zone=$ZONE --quiet
    echo "Instance deleted!"
}

function status() {
    echo "Instance status:"
    gcloud compute instances list --project=$PROJECT --filter="name=$INSTANCE"
}

function download_results() {
    echo "Downloading results from GCS bucket..."
    gsutil -m rsync -r $BUCKET/temporal-awareness/results/ results/
    echo "Results downloaded!"
}

function upload_code() {
    echo "Uploading temporal-awareness code to GCS bucket..."

    # Get the project root (parent of scripts/gcp)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

    # Upload essential files only (exclude venvs, large files)
    gsutil -m rsync -r \
        -x "venv/.*|\.venv/.*|spd_repo/.*|__pycache__/.*|\.git/.*|results/checkpoints/.*\.pkl" \
        "$PROJECT_ROOT" $BUCKET/temporal-awareness/

    echo "Code uploaded!"
}

function run_validation() {
    echo "Running probe validation on GCP..."
    gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE --command="
        cd /home/*/temporal-awareness || cd temporal-awareness
        pip install -q torch transformers scikit-learn tqdm
        python scripts/validate_probes_gcp.py --gpu
    "
}

function view_bucket() {
    echo "Contents of GCS bucket:"
    gsutil ls -r $BUCKET/temporal-awareness/
}

function setup_instance() {
    echo "Setting up instance with code from GCS..."
    gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE --command="
        # Download code from bucket
        mkdir -p temporal-awareness
        gsutil -m rsync -r $BUCKET/temporal-awareness/ temporal-awareness/
        cd temporal-awareness

        # Install dependencies
        pip install -q torch transformers scikit-learn tqdm numpy pandas

        echo 'Setup complete! Run: python scripts/validate_probes_gcp.py --gpu'
    "
}

function help() {
    echo "GCP Manager for Temporal Awareness Experiments"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  create        Create GCP instance with T4 GPU"
    echo "  ssh           SSH into instance"
    echo "  delete        Delete instance (IMPORTANT: do this when done!)"
    echo "  status        Check instance status"
    echo "  upload        Upload code to GCS bucket"
    echo "  setup         Setup instance with code from GCS"
    echo "  run           Run probe validation on GCP"
    echo "  download      Download results from GCS"
    echo "  bucket        View GCS bucket contents"
    echo "  help          Show this help"
    echo ""
    echo "Quick workflow:"
    echo "  1. $0 upload         # Upload latest code"
    echo "  2. $0 create         # Create instance"
    echo "  3. $0 setup          # Setup code on instance"
    echo "  4. $0 run            # Run validation"
    echo "  5. $0 download       # Download results"
    echo "  6. $0 delete         # Delete instance to stop charges"
    echo ""
    echo "Or interactive:"
    echo "  1. $0 upload && $0 create"
    echo "  2. $0 ssh"
    echo "  3. (on instance) python scripts/validate_probes_gcp.py --gpu"
    echo "  4. $0 download && $0 delete"
}

# Main
case "$1" in
    create)
        create_instance
        ;;
    ssh)
        ssh_instance
        ;;
    delete)
        delete_instance
        ;;
    status)
        status
        ;;
    download)
        download_results
        ;;
    upload)
        upload_code
        ;;
    setup)
        setup_instance
        ;;
    run)
        run_validation
        ;;
    bucket)
        view_bucket
        ;;
    help|*)
        help
        ;;
esac
