from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('TestCSC')
config_yaml = "test_error_judge.yaml"
# ex.add_source_file("./configs/%s" %config_yaml)
observer_mongo = MongoObserver.create(url='', db_name='db')
ex.observers.append(observer_mongo)

#加载训练参数
args = {1:2}
# 超参数设置
ex.add_config(args)

@ex.automain
def main():
    print(1)

if __name__ == '__main__':
    main()
