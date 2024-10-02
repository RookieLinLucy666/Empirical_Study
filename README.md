# Setting
```
--iid:  1 -> iid
--sharded: True -> shard method
--backdoor: True -> backdoor attack
--if_rapid_retrain: True -> rapidretrain
--if_retrain: True -> retrain
--skip_retrain: True -> skip
--forget_client_idx: [2] or [2,3,7]
--skip_FL_unlearn: True -> skip
--skip_FL_train: True -> skip

```

# Run
```python Fed_Unlearn_main.py --iid 1 --data_name cifar10 --backdoor --sharded 0 --forget_client_idx 2 3 7```