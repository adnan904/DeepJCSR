num_traces=100
num_flows=(50 100 500 1000)
input_traces=("coflows/abilene/1000-0.1-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.2-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.3-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.4-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.5-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.6-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.7-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.8-20-50.0-Z-Z-100.txt" "coflows/abilene/1000-0.9-20-50.0-Z-Z-100.txt")
echo "Started Creating Traces"
for input_file in "${input_traces[@]}";
do
    for num_flow in "${num_flows[@]}";
    do
        for ((i=100;i<100+$num_traces;i++));
        do
            python3 flow-generator.py --tf $input_file  --nf $num_flow --seed $i --res "flows/$num_flow"
        done
    done
done

wait
echo "Finished created Traces"