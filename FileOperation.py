# encoding: utf-8
import os  

def appendPath(path):  
    # 所有文件  
    fileList = []  
    # 返回一个列表，其中包含在目录条目的名称(google翻译)  
    files = os.listdir(path)  

    for f in files:  
		if(os.path.isdir(path + '/' + f)):  
            # 排除隐藏文件夹。因为隐藏文件夹过多  
			if(f[0] == '.'):  
				pass  
			else:  
                # 添加非隐藏文件夹  
                #dirList.append(path + '/' + f)
				tmpList = appendPath(path + '/' + f)
				fileList.append(tmpList)
		if(os.path.isfile(path + '/' + f)):  
            # 添加文件  
			fileList.append(path + '/' + f)
			print path + '/' + f

    return fileList

if __name__ == '__main__':  
    appendPath("train_data")
