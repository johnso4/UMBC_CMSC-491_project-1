import twitter

class TwitterAnalyzer(object):
	
	def __init__(self, credFilename):
		self.credFilename = credFilename
		self.creds = {'CONSUMER_KEY':'', 'CONSUMER_SECRET':'', 'OAUTH_TOKEN':'', 'OAUTH_TOKEN_SECRET':''}

	def connectToDB(self):
		return

	"""
	@author: Dina S.
	openCredFile opens a local file containing your twitter credentials.
	it returns a dictionary with auth cred. as key:value pairs.
	place your credentials file in the same dir. with project_1.py
	credentials MUST be in the following order in the file:
	CONSUMER_KEY
	CONSUMER_SECRET
	OAUTH_TOKEN
	OAUTH_TOKEN_SECRET

	example file contents: no need for quotes around creditials
	1a2c3b
	v6e24gg
	mdfgdsg
	4g5ys5h
	"""

	def readCredFile(self):
		
		f = open(self.credFilename, "r")
		lines = f.readlines()

		self.creds['CONSUMER_KEY'] = lines[0] 
		self.creds['CONSUMER_SECRET'] = lines[1]
		self.creds['OAUTH_TOKEN'] = lines[2]
		self.creds['OAUTH_TOKEN_SECRET'] = lines[3]

		return 

	"""
		connectToTwitter() creates a connection using dict containing
		your Twitter API credentials
	"""
	def connectToTwitter(self):

		self.readCredFile()

		print self.creds

	

def main():

	your_tw_credentials_filename = "secret_key.txt"

	tw = TwitterAnalyzer(your_tw_credentials_filename)
	tw.connectToTwitter()
	
if __name__ == "__main__":
    main()