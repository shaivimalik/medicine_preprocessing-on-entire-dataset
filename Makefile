MARKDOWN_FILES := $(wildcard markdowns/*.md)

NOTEBOOK_FILES := $(patsubst markdowns/%.md,notebooks/%.ipynb,$(MARKDOWN_FILES))

all: install-mermaid-filter $(NOTEBOOK_FILES)

notebooks/%.ipynb: markdowns/%.md
	pandoc --filter mermaid-filter --resource-path=notebooks/ $< -o $@

clean:
	rm -f $(NOTEBOOK_FILES)

install-mermaid-filter:
	@npm install --global mermaid-filter