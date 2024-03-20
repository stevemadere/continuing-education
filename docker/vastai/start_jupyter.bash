export NOTEBOOK_DIR="${NOTEBOOK_DIR:-$HOME/notebooks}"
mkdir -p "${NOTEBOOK_DIR}"
cd "${NOTEBOOK_DIR}"
export JUP_TOKEN="${JUP_TOKEN:-$(openssl rand -hex 20)}"
echo "JUP_TOKEN IS ${JUP_TOKEN}"
echo "Jupyter URL is http://localhost:8080/lab?token=${JUP_TOKEN}" | tee jupyter_url.txt 
jupyter lab --no-browser --port 8080 --IdentityProvider.token="$JUP_TOKEN" --notebook-dir="$NOTEBOOK_DIR" --allow-root
