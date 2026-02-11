# 📚 Daily Learning Hub

> Curriculum-based micro-lessons generated automatically using AI

An automated learning system that generates structured, curriculum-driven lessons on advanced topics including Natural Language Processing, Causal Inference, and Stochastic Programming. Lessons are created every 4 hours using Google's Gemini AI and published in both Markdown and HTML formats.

[![Lesson Scheduler](https://github.com/AhmetY21/TIL/actions/workflows/scheduler.yml/badge.svg)](https://github.com/AhmetY21/TIL/actions/workflows/scheduler.yml)

## 🌟 Features

- **🤖 Automated Content Generation**: Uses Google Gemini AI to generate high-quality, structured lessons
- **📖 Multiple Curricula**: Supports multiple subject areas with dedicated curriculum files
- **⏰ Scheduled Updates**: Automatic lesson generation every 4 hours via GitHub Actions
- **📱 Dual Formats**: Each lesson available in both Markdown (`.md`) and HTML (`.html`)
- **🎯 Structured Learning**: Topics follow a curriculum with prerequisites and difficulty levels
- **🌐 Web Interface**: Beautiful, responsive learning hub with organized lesson listings
- **📊 Progress Tracking**: State management to track curriculum progress for each subject

## 📋 Current Subjects

### 🗣️ Natural Language Processing
- **100+ topics** covering tokenization, embeddings, transformers, BERT, GPT, and the full NLP pipeline
- From basics to advanced concepts
- [View NLP Hub](hubs/nlp-index.html)

### 🔬 Causal Inference
- **44 topics** including DAGs, potential outcomes, propensity scores, DiD, IV, and modern ML-based causal methods
- [View Causal Inference Hub](hubs/causal-inference-index.html)

### 🎲 Stochastic Programming
- **43 topics** on modeling uncertainty, scenario trees, and decomposition methods (Benders, PHA, SDDP)
- [View Stochastic Programming Hub](hubs/stochastic-programming-index.html)

## 🏗️ Repository Structure

```
TIL/
├── curriculums/                    # Curriculum definitions
│   ├── curriculum_nlp.json
│   ├── curriculum_causal.json
│   └── curriculum_stochastic-programming.json
├── topic/                          # Generated lessons organized by subject
│   ├── nlp/
│   │   └── week_N/
│   │       └── day_YYYY-MM-DD/
│   │           └── lesson_N/
│   │               ├── topic-name.md
│   │               └── topic-name.html
│   ├── causal-inference/
│   └── stochastic-programming/
├── hubs/                           # Subject-specific index pages
│   ├── nlp-index.html
│   ├── causal-inference-index.html
│   └── stochastic-programming-index.html
├── index.html                      # Main landing page
├── generate_lesson.py              # Core lesson generator script
├── requirements.txt                # Python dependencies
└── .github/workflows/
    └── scheduler.yml               # GitHub Actions workflow

```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AhmetY21/TIL.git
   cd TIL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.0-flash  # Optional, defaults to gemini-2.0-flash
   ```

### Usage

#### Generate a Single Lesson

Run the lesson generator with a specific curriculum:

```bash
python generate_lesson.py --curriculum curriculums/curriculum_nlp.json
```

This will:
1. Read the next topic from the curriculum
2. Generate lesson content using Gemini AI
3. Save the lesson in both Markdown and HTML formats
4. Update the subject's index page
5. Advance the curriculum state

#### Generate Lessons for All Subjects

```bash
python generate_lesson.py --curriculum curriculums/curriculum_nlp.json
python generate_lesson.py --curriculum curriculums/curriculum_causal.json
python generate_lesson.py --curriculum curriculums/curriculum_stochastic-programming.json
```

## 📚 Curriculum Structure

Each curriculum is defined in a JSON file with the following structure:

```json
{
  "meta": {
    "subject": "Natural Language Processing",
    "slug": "nlp",
    "subtitle": "A curriculum-based journey into NLP",
    "prompt_domain": "Natural Language Processing"
  },
  "resources": {
    "resource_id": {
      "title": "Resource Title",
      "short": "Short Name"
    }
  },
  "topics": [
    {
      "name": "Topic Name",
      "prerequisites": ["Previous Topic"],
      "difficulty": "beginner|intermediate|advanced",
      "readings": [
        {
          "resource": "resource_id",
          "anchor": "relevant section"
        }
      ]
    }
  ]
}
```

### Curriculum Fields

- **meta**: Metadata about the subject
  - `subject`: Full name of the subject
  - `slug`: URL-friendly identifier
  - `subtitle`: Brief description
  - `prompt_domain`: Domain context for AI prompts

- **resources**: Optional learning resources and references

- **topics**: Array of topics in learning order
  - `name`: Topic title (used for lesson generation)
  - `prerequisites`: List of prerequisite topics
  - `difficulty`: Beginner, intermediate, or advanced
  - `readings`: Suggested reading materials

## 🎨 Lesson Format

Each generated lesson follows a consistent structure:

### Markdown Front Matter
```yaml
---
title: "Topic Title"
date: "YYYY-MM-DD"
week: 7
lesson: 3
slug: "topic-slug"
---
```

### Content Structure
1. **Formal Definition**: What the concept is and how it's used
2. **Application Scenario**: Real-world use case
3. **Python Method**: Code example (when applicable)
4. **Follow-up Question**: Thought-provoking question for deeper learning

## ⚙️ GitHub Actions Workflow

The repository uses GitHub Actions for automated lesson generation:

- **Schedule**: Runs every 4 hours (`0 */4 * * *`)
- **Workflow**: `.github/workflows/scheduler.yml`
- **Process**:
  1. Lists all curriculum files
  2. Generates lessons for each curriculum in parallel
  3. Creates a Pull Request with new content
  4. Auto-merges the PR

### Required Secrets

Set up the following secret in your GitHub repository:
- `GEMINI_API_KEY`: Your Google Gemini API key

## 🛠️ Customization

### Adding a New Curriculum

1. Create a new curriculum JSON file in `curriculums/`:
   ```bash
   curriculums/curriculum_your-topic.json
   ```

2. Define the curriculum structure with meta, resources, and topics

3. The workflow will automatically detect and process the new curriculum

### Adjusting Lesson Frequency

Edit the `LESSON_EVERY_HOURS` variable in `generate_lesson.py`:

```python
LESSON_EVERY_HOURS = 4  # 4 => 6 lessons/day
```

Current setting generates 6 lessons per day (every 4 hours).

### Customizing Lesson Prompts

Modify the `build_prompt()` function in `generate_lesson.py` to change the lesson structure or add additional sections.

## 📖 Viewing Lessons

### Via Web Browser

1. Open `index.html` in a web browser (or host via GitHub Pages)
2. Navigate to a subject hub
3. Browse and read lessons

### Via Files

- **Markdown**: `topic/{subject}/week_N/day_YYYY-MM-DD/lesson_N/topic-name.md`
- **HTML**: Same path with `.html` extension

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Add New Curricula**: Create curriculum files for new subjects
2. **Improve Lesson Templates**: Enhance the prompt structure
3. **UI Improvements**: Improve the web interface styling
4. **Bug Fixes**: Report and fix issues
5. **Documentation**: Improve documentation and examples

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- **Google Gemini AI**: For powering the lesson generation
- **Curriculum Sources**: Various academic resources and courses that inspired the curriculum structures
- Educational content is generated for learning purposes and may require verification for accuracy

## 📧 Contact

For questions or suggestions, please open an issue in this repository.

---

**Note**: This is an experimental learning project. While the AI-generated content is structured and educational, it should be used as a supplementary learning resource alongside traditional materials.
