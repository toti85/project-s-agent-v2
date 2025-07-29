# Project-S V2 - Quick Start Guide

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/toti85/project-s-agent-v2.git
cd project-s-agent-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# DEEPSEEK_API_KEY=your_key_here
# etc.
```

### 4. Test Installation
```bash
python test_system.py
```

### 5. Run the Agent
```bash
# Basic mode
python main.py

# Advanced complex task mode
python complex_task_tester.py
```

## 🎯 Example Tasks

### Browser Automation
```
"Navigate to Gmail and send an email to john@example.com"
```

### Web Research
```
"Research the latest AI trends and create a summary"
```

### Code Generation
```
"Create a simple calculator in Python"
```

### System Optimization
```
"Check my Windows startup programs and suggest optimizations"
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed all dependencies
2. **API Errors**: Check your API keys in the .env file
3. **Browser Issues**: Install Chrome/Chromium for browser automation

### Getting Help

- Check the main README.md for detailed documentation
- Look at example tasks in the code
- Create an issue on GitHub for bugs

## 📁 Project Structure

```
project-s-agent-v2/
├── main.py                    # Main entry point
├── complex_task_tester.py     # Advanced task processor
├── core/                      # Core system components
├── tools/                     # Tool implementations
├── integrations/              # AI model integrations
├── config/                    # Configuration files
└── utils/                     # Utility functions
```
