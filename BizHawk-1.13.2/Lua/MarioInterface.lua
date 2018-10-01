BoxRadius = 6
InputSize = (BoxRadius*2+1)*(BoxRadius*2+1)

function getPositions()
	marioX = memory.read_s16_le(0x94)
	marioY = memory.read_s16_le(0x96)

	local layer1x = memory.read_s16_le(0x1A);
	local layer1y = memory.read_s16_le(0x1C);

	screenX = marioX-layer1x
	screenY = marioY-layer1y
end

function getTile(dx, dy)
	x = math.floor((marioX+dx+8)/16)
	y = math.floor((marioY+dy)/16)
	return memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10)
end

function getSprites()
	local sprites = {}
	for slot=0,11 do
		local status = memory.readbyte(0x14C8+slot)
		if status ~= 0 then
			spritex = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256
			spritey = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256
			sprites[#sprites+1] = {["x"]=spritex, ["y"]=spritey}
		end
	end
	return sprites
end

function getExtendedSprites()
	local extended = {}
	for slot=0,11 do
		local number = memory.readbyte(0x170B+slot)
		if number ~= 0 then
			spritex = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256
			spritey = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256
			extended[#extended+1] = {["x"]=spritex, ["y"]=spritey}
		end
	end
	return extended
end

function getInputs()
        getPositions()
		emu.frameadvance()
        sprites = getSprites()
        extended = getExtendedSprites()
		emu.frameadvance()
        local inputs = {}

        for dy=-BoxRadius*16,BoxRadius*16,16 do
                for dx=-BoxRadius*16,BoxRadius*16,16 do
                        inputs[#inputs+1] = 0

                        tile = getTile(dx, dy)
                        if tile == 1 and marioY+dy < 0x1B0 then
                                inputs[#inputs] = 1
                        end

                        for i = 1,#sprites do
                                distx = math.abs(sprites[i]["x"] - (marioX+dx))
                                disty = math.abs(sprites[i]["y"] - (marioY+dy))
                                if distx <= 8 and disty <= 8 then
                                        inputs[#inputs] = -1
                                end
                        end

                        for i = 1,#extended do
                                distx = math.abs(extended[i]["x"] - (marioX+dx))
                                disty = math.abs(extended[i]["y"] - (marioY+dy))
                                if distx < 8 and disty < 8 then
                                        inputs[#inputs] = -1
                                end
                        end
                end
        end

        --mariovx = memory.read_s8(0x7B)
        --mariovy = memory.read_s8(0x7D)

        return inputs
end

clock = os.clock
t0 = clock()
furthestRight = marioX
timeoutCounter = 0
local socket = require("socket")
local host, port = "127.0.0.1", 10309
local tcp = assert(socket.tcp())
tcp:connect(host,port)
tcp:settimeout(0)
function Export()
			t0 = clock()
			if tcp then
				returnString = ""
				inputs = getInputs()
				inputs[#inputs+1] = marioX - startingX
				if Died() then
					inputs[#inputs+1] = 1
					savestate.loadslot(0)
					furthestRight = 0
				else
					inputs[#inputs+1] = 0
				end
				for index, value in ipairs(inputs) do
					returnString = returnString .. value
				end
				tcp:send(returnString)
			end
end
--return true if not moving
function Died()
	if furthestRight >= marioX then
		timeoutCounter = timeoutCounter + 1
		if timeoutCounter > 10 then
			timeoutCounter = 0
			return true
		end
	end
	if marioX > furthestRight then
		furthestRight = marioX
		timeoutCounter = 0
	end
	return false
end
savestate.saveslot(0)
startingX = marioX
while true do
Export()

emu.frameadvance();
end
tcp:close()
