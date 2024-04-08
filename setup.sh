mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
[client]\n\
showErrorDetails = false\n\
" >>~/.streamlit/config.toml