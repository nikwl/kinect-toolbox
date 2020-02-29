import socket

from imagezmq import imagezmq

class Bouncer():
	def connect_to_incoming(self, incoming_socket=None):
		if incoming_socket:
			self.imageHub = imagezmq.ImageHub(connect_to=incoming_socket)
		else:
			self.imageHub = imagezmq.ImageHub()

		print("Connected to incoming image stream")

	def connect_to_outgoing(self, outgoing_addr, outgoing_socket=None):
		if outgoing_socket:
			self.sender = imagezmq.ImageSender(connect_to="tcp://{}:"+outgoing_socket.format(outgoing_addr))
		else:
			self.sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(outgoing_addr))
		self.sender_address = socket.gethostname()

		print("Connected to outgoing image stream")

	def get_image(self):
		_, frame = self.imageHub.recv_image()
		self.imageHub.send_reply(b'OK')
		return frame

	def send_image(self, frame):
		self.sender.send_image(self.sender_address, frame)
