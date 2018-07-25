import codecs
import sys
import spacy

def transformer(data_in, data_out, vocab_words, vocab_tags, vocab_deps):
    id2tokens = {}
    tokens2id = {}
    id2tags = {}
    tags2id = {}
    id2deps = {}
    deps2id = {}
    with codecs.open(vocab_words, "r") as f1:
        for line in f1.readlines():
            token, id = line.strip().split("##")
            id = int(id)
            id2tokens[id] = token
            tokens2id[token] = id

    tokens_count = len(tokens2id)
    print("Load words vocab done!")

    with codecs.open(vocab_tags, "r") as f4:
        for line in f4.readlines():
            tag, id = line.strip().split("##")
            id = int(id)
            id2tags[id] = tag
            tags2id[tag] = id

    tags_count = len(tags2id)
    print("Load Tags vocab done!")

    with codecs.open(vocab_deps, 'r') as f5:
        for line in f5.readlines():
            dep, id = line.strip().split("##")
            id = int(id)
            id2deps[id] = dep
            deps2id[dep] = id

    deps_count = len(deps2id)
    print("Load Dependency vocab done!")

    with codecs.open(data_in, 'r') as f2:
        with codecs.open(data_out, 'w') as f3:
            nlp = spacy.load('en')
            for line in f2.readlines():
                line = line.strip()
                tokens = line.split()

                for token in tokens:
                    id = tokens2id.get(token, tokens2id["<unk>"])
                    f3.write(str(id) + ' ')
                f3.write('#' + ' ')

                doc = nlp(line)
                for token in doc:
                    tag = token.tag_
                    tagid = tags2id.get(tag, tags2id["<unk>"])
                    f3.write(str(tagid) + ' ')
                f3.write('#' + ' ')

                for token in doc:
                    dep = token.dep_
                    depid = deps2id.get(dep, deps2id["<unk>"])
                    f3.write(str(depid) + ' ')
                f3.write('#' + ' ')

                for token in doc:
                    head = token.head.text
                    headid = tokens2id.get(head, tokens2id["<unk>"])
                    f3.write(str(headid) + ' ')

                f3.write("\n")

    print("\nTokens Number:", tokens_count, "\n")
    print("Tags Nmber:", tags_count, "\n")
    print("Dependency Number:", deps_count)

if __name__ == '__main__':
    args = sys.argv
    data_in = args[1]
    data_out = args[2]
    vocab_words = args[3]
    vocab_tags = args[4]
    vocab_deps = args[5]
    transformer(data_in, data_out, vocab_words, vocab_tags, vocab_deps)


