import os

os.system('tar -zxvf ./zipdata/ta1-cadets-e3-official.json.tar.gz -C ./unziped_data')
os.system('tar -zxvf ./zipdata/ta1-cadets-e3-official-2.json.tar.gz -C ./unziped_data')
os.system('tar -zxvf ./zipdata/ta1-fivedirections-e3-official-2.json.tar.gz -C ./unziped_data')
os.system('tar -zxvf ./zipdata/ta1-theia-e3-official-1r.json.tar.gz -C ./unziped_data')
os.system('tar -zxvf ./zipdata/ta1-theia-e3-official-6r.json.tar.gz -C ./unziped_data')
os.system('tar -zxvf ./zipdata/ta1-trace-e3-official-1.json.tar.gz -C ./unziped_data')

# # 保存文件名path
# path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json',
#              'ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']
#
# os.system('cp ta1-theia-e3-official-1r.json.txt ./data/theia_train.txt')
# os.system('cp ta1-theia-e3-official-6r.json.8.txt ./data/theia_test.txt')
# os.system('cp ta1-cadets-e3-official.json.1.txt ./data/cadets_train.txt')
# os.system('cp ta1-cadets-e3-official-2.json.txt ./data/cadets_test.txt')
# os.system(
#     'cp ta1-fivedirections-e3-official-2.json.txt ./data/fivedirections_train.txt')
# os.system(
#     'cp ta1-fivedirections-e3-official-2.json.23.txt ./data/fivedirections_test.txt')
# os.system('cp ta1-trace-e3-official-1.json.txt ./data/trace_train.txt')
# os.system('cp ta1-trace-e3-official-1.json.4.txt ./data/trace_test.txt')
#
# os.system('rm ta1-*')