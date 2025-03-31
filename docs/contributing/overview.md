# 💻 Contributing: Join the Preprocessing Revolution! 🛠️

<div class="intro-container">
  <div class="intro-content">
    <h2>Help us build the future of data preprocessing!</h2>
    <p>We're excited to welcome new contributors to KDP. This guide will help you get started on your contribution journey.</p>
  </div>
</div>

## 🏁 Contribution Process Overview

<div class="steps-container">
  <div class="step-card">
    <div class="step-header">
      <span class="step-number">1</span>
      <h3>Getting Started</h3>
    </div>
    <div class="step-content">
      <ul class="feature-list">
        <li><strong>Fork the Repository</strong>: Visit our GitHub page, fork the repository, and clone it to your local machine.</li>
        <li><strong>Set Up Your Environment</strong>: Make sure you have TensorFlow, Loguru, and all necessary dependencies installed.</li>
        <li><strong>Install Pre-commit Hook</strong>: This ensures code quality before commits.</li>
      </ul>
      <div class="code-container">

```bash
# Install pre-commit hook
conda install -c conda-forge pre-commit

# Go to the root folder of the repository and run
pre-commit install
```

      </div>
    </div>
  </div>

  <div class="step-card">
    <div class="step-header">
      <span class="step-number">2</span>
      <h3>Making Changes</h3>
    </div>
    <div class="step-content">
      <ul class="feature-list">
        <li><strong>Create a Branch</strong>: Always work in a dedicated branch for your changes.</li>
        <li><strong>Follow Coding Standards</strong>: Maintain the project's coding style and conventions.</li>
        <li><strong>Write Tests</strong>: All new features must include tests.</li>
        <li><strong>Use Standardized Commit Messages</strong>: Follow the format below.</li>
      </ul>
      <div class="code-container">

```
{LABEL}(KDP): {message}

# Examples:
feat(KDP): Add distribution-aware encoding
fix(KDP): Resolve memory leak in feature selection
```

      </div>
    </div>
  </div>

  <div class="step-card">
    <div class="step-header">
      <span class="step-number">3</span>
      <h3>Submitting Your Work</h3>
    </div>
    <div class="step-content">
      <ul class="feature-list">
        <li><strong>Create Small MRs</strong>: Keep merge requests under 400 lines for easier review.</li>
        <li><strong>Request Code Review</strong>: All code must be reviewed before merging.</li>
        <li><strong>Address Feedback</strong>: Resolve all comments and ensure CI checks pass.</li>
        <li><strong>Tests Must Pass</strong>: <span class="highlight">NO TESTS = NO MERGE 🚨</span></li>
      </ul>
    </div>
  </div>
</div>

## 💡 Feature Requests & Issues

<div class="feature-showcase">
  <div class="feature-header">
    <h3>Have ideas or found a bug?</h3>
  </div>
  <div class="feature-content">
    <p>We welcome your input! Please use our GitHub issues page to:</p>
    <ul class="feature-list">
      <li>Report bugs or unexpected behavior</li>
      <li>Suggest new features or improvements</li>
      <li>Discuss implementation approaches</li>
    </ul>
    <a href="https://github.com/piotrlaczkowski/keras-data-processor/issues" class="action-button">Open an Issue</a>
  </div>
</div>

## 📝 Commit Message Guidelines

<div class="examples-container">
  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">🏷️</span>
      <h3>Label Types</h3>
    </div>
    <div class="example-content">
      <div class="table-container">
        <table>
          <tr>
            <th>Label</th>
            <th>Usage</th>
            <th>Version Impact</th>
          </tr>
          <tr>
            <td><code>break</code></td>
            <td>Changes that break backward compatibility</td>
            <td>major</td>
          </tr>
          <tr>
            <td><code>feat</code></td>
            <td>New backward-compatible features</td>
            <td>minor</td>
          </tr>
          <tr>
            <td><code>fix</code></td>
            <td>Bug fixes</td>
            <td>patch</td>
          </tr>
          <tr>
            <td><code>docs</code></td>
            <td>Documentation changes</td>
            <td>patch</td>
          </tr>
          <tr>
            <td><code>style</code></td>
            <td>Code style changes (formatting, etc.)</td>
            <td>patch</td>
          </tr>
          <tr>
            <td><code>refactor</code></td>
            <td>Code changes that neither fix bugs nor add features</td>
            <td>patch</td>
          </tr>
          <tr>
            <td><code>perf</code></td>
            <td>Performance improvements</td>
            <td>patch</td>
          </tr>
          <tr>
            <td><code>test</code></td>
            <td>Adding or updating tests</td>
            <td>minor</td>
          </tr>
          <tr>
            <td><code>build</code></td>
            <td>Build system or dependency changes</td>
            <td>patch</td>
          </tr>
          <tr>
            <td><code>ci</code></td>
            <td>CI configuration changes</td>
            <td>minor</td>
          </tr>
        </table>
      </div>
    </div>
  </div>
</div>

## 🔄 Merge Request Process

<div class="feature-showcase">
  <div class="feature-header">
    <h3>Creating Effective Merge Requests</h3>
  </div>
  <div class="feature-content">
    <p>Merge requests are the heart of our collaborative development process:</p>
    <ul class="feature-list">
      <li>Create your MR early - even as a work in progress</li>
      <li>Use the same naming convention as commits: <code>{LABEL}(KDP): {message}</code></li>
      <li>Break large features into smaller, focused MRs</li>
      <li>Include relevant tests for your changes</li>
      <li>Ensure all CI checks pass before requesting review</li>
      <li>Address all feedback before merging</li>
    </ul>
    <div class="tip-box">
      <span class="tip-icon">💡</span>
      <p>Merge requests generate our changelog automatically, so clear and descriptive messages help everyone understand your contributions!</p>
    </div>
  </div>
</div>

<style>
/* Base styling */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #333;
}

/* Intro section */
.intro-container {
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-radius: 10px;
  padding: 25px;
  margin: 30px 0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.intro-content h2 {
  margin-top: 0;
  color: #4a86e8;
}

/* Step cards */
.steps-container {
  display: flex;
  flex-direction: column;
  gap: 25px;
  margin: 30px 0;
}

.step-card {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.step-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.step-header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.step-number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  background-color: #4a86e8;
  color: white;
  border-radius: 50%;
  margin-right: 15px;
  font-weight: bold;
}

.step-header h3 {
  margin: 0;
  color: #333;
}

.step-content {
  padding: 20px;
}

/* Feature list */
.feature-list {
  padding-left: 20px;
  margin-bottom: 20px;
}

.feature-list li {
  margin-bottom: 10px;
}

/* Code containers */
.code-container {
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  margin-top: 15px;
}

.code-container pre {
  margin: 0;
  padding: 15px;
}

/* Feature showcase */
.feature-showcase {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  margin: 30px 0;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-showcase:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.feature-header {
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.feature-header h3 {
  margin: 0;
  color: #333;
}

.feature-content {
  padding: 20px;
}

/* Examples section */
.examples-container {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  margin: 30px 0;
}

.example-card {
  background-color: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.example-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.example-header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e9ecef 100%);
  border-bottom: 1px solid #e9ecef;
}

.example-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.example-header h3 {
  margin: 0;
  color: #333;
}

.example-content {
  padding: 20px;
}

/* Table styling */
.table-container {
  overflow-x: auto;
  margin: 15px 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 15px 0;
}

th {
  background-color: #f0f7ff;
  text-align: left;
  padding: 12px 15px;
  border-bottom: 2px solid #4a86e8;
}

td {
  padding: 10px 15px;
  border-bottom: 1px solid #e9ecef;
}

tr:nth-child(even) {
  background-color: #f8f9fa;
}

tr:hover {
  background-color: #f0f7ff;
}

/* Buttons */
.action-button {
  display: inline-block;
  background-color: #4a86e8;
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  text-decoration: none;
  font-weight: 500;
  margin-top: 15px;
  transition: background-color 0.3s ease;
}

.action-button:hover {
  background-color: #3a76d8;
}

/* Tip box */
.tip-box {
  background-color: #f0f7ff;
  border-left: 4px solid #4a86e8;
  padding: 15px;
  margin-top: 20px;
  border-radius: 0 5px 5px 0;
  display: flex;
  align-items: flex-start;
}

.tip-icon {
  font-size: 1.5em;
  margin-right: 15px;
}

.highlight {
  background-color: #ffecb3;
  padding: 2px 5px;
  border-radius: 3px;
  font-weight: bold;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .examples-container {
    grid-template-columns: 1fr;
  }
}
</style>
