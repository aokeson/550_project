#!/bin/bash
declare -a servers=("8806" "8807" "8808" "8809" "8810")

start() {
	python leader_nofreeze.py &
	for number in "${servers[@]}"
	do
		python worker_nofreeze.py "$number" &
	done
	sleep 20
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
