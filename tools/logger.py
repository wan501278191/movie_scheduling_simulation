import os
import datetime


# 构造一个带有时间信息的日志类
class Logger(object):
    def __init__(self, logdir, is_print=True):
        self.logdir = logdir
        self.logfile_name = 'MovieSchedulingTraining.log'
        self.is_print = is_print

    def write_log(self, message):
        # --- 核心修改：增加一个判断 ---
        # 只有当日志目录不是 'nul' 或 '/dev/null' (黑洞设备) 时，才执行文件写入操作
        if self.logdir and self.logdir != os.devnull:
            # 创建日志目录
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            # 写入日志
            log_file_path = os.path.join(self.logdir, self.logfile_name)
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

        # 打印到控制台的逻辑保持不变
        if self.is_print:
            print(message)

    def info(self, message):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[INFO] [{current_time}] {message}"
        self.write_log(log_message)

    def warning(self, message):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[WARNING] [{current_time}] {message}"
        self.write_log(log_message)

    def error(self, message):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[ERROR] [{current_time}] {message}"
        self.write_log(log_message)