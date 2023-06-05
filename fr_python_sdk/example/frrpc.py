import xmlrpc.client

def RPC(ip):
    link = 'http://' + ip + ":20003"
    s = xmlrpc.client.ServerProxy(link)
    return s
