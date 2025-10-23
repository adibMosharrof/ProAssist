# DST Unified JSON Construction - Installation Guide

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install openai pyyaml jsonschema
   ```

2. **Set OpenAI API key (optional):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   *Note: Without API key, system uses fallback DST generation*

3. **Test installation:**
   ```bash
   python custom/src/dst_data_builder/sample_test.py
   ```

4. **Run demo:**
   ```bash
   python custom/src/dst_data_builder/demo.py
   ```

## Usage

### Process all datasets:
```bash
python -m custom.src.dst_data_builder.run
```

### Process single dataset:
```bash
python -m custom.src.dst_data_builder.run --dataset assembly101
```

### Custom options:
```bash
python -m custom.src.dst_data_builder.run \
  --dataset assembly101 \
  --input-file /path/to/custom.json \
  --log-level DEBUG
```

## Output

Files are created in `custom/outputs/dst_unified/` with structure:
```
custom/outputs/dst_unified/
├── assembly101/
│   ├── video_1.json
│   └── video_2.json
├── ego4d/
└── ...
```

Each JSON file contains:
- Complete 3-level DST hierarchy
- Timestamped node states 
- Dialog turns with state snapshots
- Progress summaries
- Metadata

## Configuration

Edit `custom/config/dst_builder_config.yaml` to customize:
- Input/output paths
- GPT model settings
- Validation rules
- Summary generation
- Logging options

## Troubleshooting

- **Import errors**: Ensure you're in the ProAssist root directory
- **API errors**: Check OpenAI API key and network connection
- **Memory issues**: Reduce batch size in config
- **Validation errors**: Check logs, system will attempt auto-fix

For detailed documentation, see `custom/src/dst_data_builder/README.md`