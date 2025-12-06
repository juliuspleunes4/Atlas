# License Guide

## Project License

Atlas is licensed under the **MIT License**. See the [LICENSE](../LICENSE) file in the root directory for the full license text.

## What Does This Mean?

The MIT License is one of the most permissive open-source licenses. It allows you to:

### ✅ Permitted Uses

- **Commercial Use**: Use Atlas in commercial projects and products
- **Modification**: Modify the source code to fit your needs
- **Distribution**: Distribute original or modified versions
- **Private Use**: Use Atlas privately without sharing modifications
- **Sublicensing**: Include Atlas in projects with different licenses

### 📋 Requirements

When using Atlas, you must:

1. **Include Copyright Notice**: Keep the original copyright notice in all copies
2. **Include License Text**: Include the full MIT License text with distributions
3. **No Warranty Disclaimer**: Acknowledge that Atlas comes with no warranty

### ❌ Limitations

- **No Liability**: Authors are not liable for any damages or issues
- **No Warranty**: Software is provided "as-is" without guarantees
- **No Trademark Rights**: License doesn't grant trademark rights

## Using Atlas in Your Projects

### For Personal Projects

```python
# You can use Atlas freely in personal projects
from atlas.model import AtlasLM

model = AtlasLM(config)
# ... use the model
```

No additional steps required! Just follow the license terms.

### For Commercial Projects

You can use Atlas in commercial products:

1. Include the MIT License text
2. Keep copyright notices
3. That's it! No royalties or fees

### For Derived Works

If you modify Atlas and distribute it:

1. Include original copyright notice
2. Include MIT License text
3. Optionally note your modifications
4. You may license your modifications differently, but must preserve MIT License for original code

Example attribution:
```
This software includes code from Atlas (https://github.com/juliuspleunes4/Atlas)
Copyright (c) 2025 Julius Pleunes
Licensed under the MIT License
```

## Third-Party Dependencies

Atlas uses several third-party libraries, each with their own licenses:

### Core Dependencies

| Package | License | Type | Usage |
|---------|---------|------|-------|
| PyTorch | BSD-3-Clause | Permissive | Deep learning framework |
| NumPy | BSD-3-Clause | Permissive | Numerical computing |
| tiktoken | MIT | Permissive | Tokenization |
| PyYAML | MIT | Permissive | Configuration |
| tqdm | MIT/MPL-2.0 | Permissive | Progress bars |

All dependencies use permissive licenses compatible with commercial use.

### Verifying Dependencies

Check licenses of all dependencies:
```bash
pip install pip-licenses
pip-licenses
```

## Model Licenses

### Trained Models

When you train a model using Atlas:

- **Atlas code**: Remains MIT licensed
- **Your trained model weights**: You own these
- **Training data**: Subject to data source license

### Training Data License

If using Wikipedia SimpleEnglish dataset:
- **License**: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)
- **Requirements**: 
  - Attribute Wikipedia
  - Share derivatives under same license
  - Commercial use allowed

When using custom training data, ensure you have rights to use it.

### Model Distribution

When distributing trained models:

1. **Model Weights**: You can license these as you wish
2. **Atlas Code**: Must preserve MIT License if included
3. **Training Data**: Follow source data license terms

## Common Scenarios

### Scenario 1: Using Atlas to Train Custom Models

**Q**: I want to train a model for my startup. Do I need permission?

**A**: No! The MIT License allows commercial use. Train and use your models freely. Just keep the license notices if you distribute the Atlas code.

### Scenario 2: Modifying Atlas Code

**Q**: I improved the attention mechanism. Must I share my changes?

**A**: No! The MIT License doesn't require sharing modifications. However, contributing back helps the community and is encouraged.

### Scenario 3: Selling Atlas-Based Products

**Q**: Can I sell a product built with Atlas?

**A**: Yes! You can build and sell commercial products. Include the MIT License notice for Atlas code you distribute.

### Scenario 4: Integrating into Closed-Source Software

**Q**: Can I use Atlas in proprietary software?

**A**: Yes! The MIT License allows integration into closed-source products. Just include the license notice.

### Scenario 5: Research and Publications

**Q**: I used Atlas in my research. How do I cite it?

**A**: Please cite Atlas in your publications:

```bibtex
@software{atlas2025,
  author = {Pleunes, Julius},
  title = {Atlas: A From-Scratch Large Language Model Implementation},
  year = {2025},
  url = {https://github.com/juliuspleunes4/Atlas}
}
```

### Scenario 6: Forking Atlas

**Q**: Can I fork Atlas and create a competing project?

**A**: Yes! Fork away. Just preserve the original license and copyright notices.

## Best Practices

### Attribution

While not strictly required for code not distributed, consider:

- Citing Atlas in research papers
- Mentioning Atlas in project documentation
- Linking to the Atlas repository
- Acknowledging contributors

### Contributing Back

Consider contributing improvements:

- Bug fixes benefit everyone
- New features make Atlas better
- Documentation helps other users
- You get recognition and feedback

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### License Compliance Checklist

When distributing Atlas or derivatives:

- [ ] Include original MIT License text
- [ ] Preserve copyright notices
- [ ] Document any modifications (optional but recommended)
- [ ] Check your training data license
- [ ] Include licenses for all dependencies
- [ ] Add your own copyright for modifications (optional)

## Frequently Asked Questions

### Do I need to open-source my trained models?

No. Your model weights are yours. The MIT License doesn't require sharing them.

### Can I change the license of my fork?

You can add your own license for your modifications, but the original Atlas code must remain under MIT License.

### Do I need to mention Atlas in my product?

Only if you distribute Atlas code. If you just use it internally or only distribute trained models, attribution is appreciated but not required.

### Can I use Atlas in a patent-protected product?

The MIT License doesn't grant or restrict patent rights. Consult a lawyer if patents are involved.

### What if I modify Atlas significantly?

You still must preserve the original MIT License for Atlas code. You can add your own license for your additions.

## Need Legal Advice?

This guide is informational only and not legal advice. For legal questions:

- Consult a qualified attorney
- Review the full MIT License text
- Check licenses of dependencies
- Understand your training data rights

## Questions?

For questions about Atlas licensing:

1. Review this guide and the [LICENSE](../LICENSE) file
2. Check [FAQ.md](FAQ.md)
3. Open an issue on GitHub
4. Contact the project maintainers

---

**Summary**: Atlas uses the permissive MIT License. You can use it freely in commercial and private projects, modify it, and distribute it. Just keep the license notices.

---

**Last Updated**: December 7, 2025
