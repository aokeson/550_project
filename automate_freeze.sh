#!/bin/bash
declare -a servers=("8801" "8802" "8803" "8804" "8805" "8806" "8807" "8808" "8809" "8810")

start() {
	python leader_freeze.py &
	for number in "${servers[@]}"
	do
		python worker.py "$number" &
	done
	sleep 15
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
