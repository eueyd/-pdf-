import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import torch
from PIL import Image
import threading

class PDFQA(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.progress.emit("初始化文档向量模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )
        
        self.progress.emit("初始化问答模型...")
        model_name = "uer/roberta-base-chinese-extractive-qa"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        self.load_pdfs("source")
        
    def load_pdfs(self, pdf_dir):
        self.progress.emit("加载PDF文件...")
        try:
            all_docs = []
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(pdf_dir, filename)
                    self.progress.emit(f"处理: {filename}")
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    
            self.progress.emit("分割文档...")
            self.docs = self.text_splitter.split_documents(all_docs)
            
            self.progress.emit("创建向量数据库...")
            self.vectordb = FAISS.from_documents(self.docs, self.embeddings)
            
            self.progress.emit(f"完成! 共处理{len(self.docs)}个文本块")
            
        except Exception as e:
            self.progress.emit(f"错误: {str(e)}")
    
    def ask(self, question):
        try:
            relevant_docs = self.vectordb.similarity_search(question, k=3)
            context = " ".join([doc.page_content for doc in relevant_docs])
            
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=150
            )
            
            self.finished.emit({
                "success": True,
                "answer": result["answer"],
                "confidence": f"{result['score']:.2%}",
                "context": relevant_docs
            })
            
        except Exception as e:
            self.finished.emit({
                "success": False,
                "error": str(e)
            })

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboMaster PDF QA系统")
        self.resize(1000, 800)
        
        # 创建中心部件
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # 状态标签
        self.status = QLabel("初始化中...")
        layout.addWidget(self.status)
        
        # 问题输入区
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        self.question = QLineEdit()
        self.question.setPlaceholderText("请输入问题")
        input_layout.addWidget(self.question)
        
        ask_btn = QPushButton("提问")
        ask_btn.clicked.connect(self.on_ask)
        input_layout.addWidget(ask_btn)
        
        upload_btn = QPushButton("上传图片")
        upload_btn.clicked.connect(self.on_upload)
        input_layout.addWidget(upload_btn)
        
        layout.addWidget(input_widget)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧图片显示
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        splitter.addWidget(self.image_label)
        
        # 右侧答案显示
        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        splitter.addWidget(self.answer)
        
        # 初始化QA系统
        self.qa_thread = QThread()
        self.qa_system = PDFQA()
        self.qa_system.moveToThread(self.qa_thread)
        self.qa_system.progress.connect(self.on_progress)
        self.qa_system.finished.connect(self.on_answer)
        self.qa_thread.start()

    def on_progress(self, msg):
        self.status.setText(msg)
        
    def on_ask(self):
        question = self.question.text()
        if not question:
            return
            
        self.status.setText("处理问题中...")
        self.answer.clear()
        
        QTimer.singleShot(0, lambda: self.qa_system.ask(question))
        
    def on_answer(self, result):
        if result["success"]:
            output = f"答案: {result['answer']}\n"
            output += f"置信度: {result['confidence']}\n\n"
            output += "相关段落:\n"
            
            for i, doc in enumerate(result["context"], 1):
                output += f"\n{i}. 来源: {os.path.basename(doc.metadata['source'])}\n"
                output += f"页码: {doc.metadata['page']}\n"
                output += f"内容: {doc.page_content}\n"
                
            self.answer.setText(output)
            self.status.setText("回答完成")
        else:
            self.answer.setText(f"错误: {result['error']}")
            self.status.setText("处理出错")
        
    def on_upload(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.status.setText(f"已选择图片: {file_name}")
            pixmap = QPixmap(file_name)
            scaled = pixmap.scaled(
                400, 300,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            
    def closeEvent(self, event):
        self.qa_thread.quit()
        self.qa_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置中文字体
    font = QFont("Microsoft YaHei")
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())