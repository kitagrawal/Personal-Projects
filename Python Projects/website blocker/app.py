import time
from datetime import datetime as dt

host_tmp = "hosts"
hosts_path = "/etc/hosts" #change this for linux and Mac accordingly
redirect = "127.0.0.1" #change this to where you want your search to redirect to
#website = input("enter the URL")
#website_list.append(website)
website_list = ["www.facebook.com", "www.youtube.com", "www.gmail.com"] #list of websites to block during work hours

while True:
    if dt(dt.now().year,dt.now().month,dt.now().day,8) < dt.now() < dt(dt.now().year,dt.now().month,dt.now().day,20): #between 8 AM and 8 PM are working hours
        print("working hours")
        with open(hosts_path,'r+') as f:
            content = f.read()
            for website in website_list:
                if website in content: pass;
                else: f.write(redirect+" "+website+"\n");

    else:
        with open(hosts_path,'r+') as file: #restoring the host file to original state in off hours
            content = file.readlines()
            file.seek(0)
            for line in content:
                if not any(website in line for website in website_list):
                    file.write(line)
            file.truncate()
        print("Fun hours")
    time.sleep(300) #check every 5 mins
