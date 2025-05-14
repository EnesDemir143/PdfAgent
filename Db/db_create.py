import sqlite3

class Database:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_history_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_message TEXT,
            bot_response TEXT)''')
        self.connection.commit()

    def insert_history(self, user_message, bot_response=None):
        self.cursor.execute('''
            INSERT INTO chat_history (user_message, bot_response)
            VALUES (?, ?, ?)''', (user_message, bot_response))
        self.connection.commit()

    def fetch_history(self, chat_id):
        self.cursor.execute('''
            SELECT * FROM chat_history WHERE id = ?''', (chat_id,))
        return self.cursor.fetchall()