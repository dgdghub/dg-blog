import markdownIt from 'markdown-it'
import markdownItLatex from 'markdown-it-latex'
const mdi = markdownIt()
mdi.use(markdownItLatex)
mdi.render('`$E = mc^2$`')
mdi.render('`@(1/2[1-(1/2)^n])/(1-(1/2))=s_n@`')
mdi.render(`\`\`\`math\oint_C x^3\, dx + 4y^2\, dy\`\`\``)
mdi.render(`\`\`\`AsciiMathoint_Cx^3 dx+4y^2 dy\`\`\``)