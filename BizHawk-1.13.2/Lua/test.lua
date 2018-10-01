socket = require("socket")
host, port = "127.0.0.1", 10309
tcp = assert(socket.tcp())
tcp:connect(host,port)

while true do
	tcp:settimeout(0)
	tcp:send("Stream starting...")
end
tcp:close()
