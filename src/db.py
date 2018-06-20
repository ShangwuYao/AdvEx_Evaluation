import os
import sys
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON


PG_HOST = os.environ['PG_HOST']
PG_USERNAME = os.environ['PG_USERNAME']
PG_PASSWORD = os.environ['PG_PASSWORD']
PG_DATABASE = os.environ['PG_DATABASE']

DB_URL = 'postgresql+psycopg2://{user}:{pw}@{host}/{db}'.format(
	user=PG_USERNAME, 
	pw=PG_PASSWORD,
	host=PG_HOST,
	db=PG_DATABASE
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
db.init_app(app)


class User(db.Model):
	user_id = db.Column(db.Integer, primary_key=True)
	nickname = db.Column(db.String(200), unique=False, nullable=False)
	email = db.Column(db.String(200), unique=True, nullable=False)
	password = db.Column(db.String(200), unique=False, nullable=False)

	def __repr__(self):
		return '<User ID: {}, nickname: {}, email: {}>'.format(self.user_id, 
			self.nickname, self.email)


class Submission(db.Model):
	submission_id = db.Column(db.Integer, primary_key=True)
	model_name = db.Column(db.String(80), nullable=False)
	status = db.Column(db.String(80), nullable=False)
	s3_model_key = db.Column(db.String(80), nullable=False)
	s3_index_key = db.Column(db.String(80), nullable=False)
	created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
	feedback = db.Column(JSON, nullable=True)

	user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
	user = db.relationship('User', backref=db.backref('submissions', lazy=True), uselist=False)

	def __repr__(self):
		return '<Submission ID: {}, model_name: {}, status: {}, model_key: {}, index_key: {}, created_at: {}>'\
			.format(self.submission_id, self.model_name, self.status, self.s3_model_key, self.s3_index_key, self.created_at)


def init_db():
	print('Drop')
	db.drop_all()
	print('Create')
	db.create_all()

	user = User(
		nickname='Dave',
		email='dave@gmail.com',
		password='aircrash'
	)

	submission = Submission(
		user_id=1,
		model_name='VGG-16 v1.0',
		status='submitted',
		s3_model_key='model.h5',
		s3_index_key='index.json'
	)

	db.session.add(user)
	db.session.add(submission)
	db.session.commit()


def test_alpha():
	feedback = Submission.query.get(1).feedback
	print(feedback if feedback else 'EMPTY FEEDBACK')


if __name__ == '__main__':
	if sys.argv[1] == 'init':
		init_db()
	if sys.argv[1] == 'test':
		test_alpha()
