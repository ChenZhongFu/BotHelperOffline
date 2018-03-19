import paramiko
import os

def trans_model(local_path,appId):
    ip="111.207.243.70"
    port=22
    username='root'
    password='cYz@1Q#2s%3C!'


    remote_path="/home/cyzhang/classify"+appId+".h5"

    trans = paramiko.Transport((ip,port))
    trans.connect(username=username,password=password)

    sftp = paramiko.SFTPClient.from_transport(trans)


    sftp.put(localpath=local_path,remotepath=remote_path)

    trans.close()
    print("ftp over")
