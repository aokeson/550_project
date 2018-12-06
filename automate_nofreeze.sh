#!/bin/bash
declare -a servers=("8801" "8802" "8803" "8804" "8805")

start() {
	python leader_nofreeze.py &
	for number in "${servers[@]}"
	do
		python worker.py "$number" &
	done
	sleep 15
	python client_nofreeze.py &
}

stop() {
	for number in "${servers[@]}"
	do
		kill $(lsof -t -i :"$number");
	done
	pkill -f "leader_nofreeze.py"
}

"$@"
