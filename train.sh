if [ $(dpkg-query -W -f='${Status}' pip 2>/dev/null | grep -c "ok installed") -eq 0 ];
then
  sudo apt install pip
fi
pip install -r requirements.txt
python train.py \
--config_file train \