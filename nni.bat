pip install nni

echo Step 1: obtaining ngrok
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip

echo Step 2: unzipping ngrok file
unzip ngrok-stable-linux-amd64.zip

echo Step 3: Creating nni_repo directory
mkdir -p nni_repo

echo Step 4: clone NNI's offical repo
git clone https://github.com/microsoft/nni.git nni_repo/nni 


./ngrok authtoken 2GlnNbXOddav5gf2NqwKi4GLcTm_3Q7vEZCZ5pcmBguxG3Huk

nnictl create --config nni_repo/nni/examples/trials/mnist-pytorch/config.yml --port 5000 & 

curl -s http://localhost:4040/api/tunnels
echo In the line above, a public URL has been created http://_______.ngrok.io

@REM nnictl stop --all