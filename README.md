# Olimpus-gymMicroRTS

An experiment with RL in RTS games

By Fabián Cid

## How to install

Follow the next steps:

Get base repo
> git clone --recursive https://github.com/vwxyzjn/gym-microrts.git

Open the folder
> cd gym-microrts

Install JDK (in Linux)
> sudo add-apt-repository ppa:openjdk-r/ppa
> sudo apt install openjdk-8-jdk
> sudo apt install openjdk-8-source

Install Poetry
> sudo pip install poetry

Build gym-microRTS
> sudo bash build.sh
> sudo pip install gym-microrts

Change to #71 version
> git checkout 6d3644b

Execute the example
> python3 hello_word.py
