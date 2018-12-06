#!/bin/bash
declare -a servers=("8801" "8802" "8803" "8804" "8805")

start() {
	python leader_freeze.py &
	for number in "${servers[@]}"
	do
		python worker_freeze.py "$number" &
	done
	sleep 20
	python client_freeze.py &
}

stop() {
	for number in "${servers[@]}"
	do
		kill $(lsof -t -i :"$number");
	done
	pkill -f "leader_freeze.py"
}

"$@"
