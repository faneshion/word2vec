#include<iostream>
#include<string>
#include<vector>
#include<numeric>
#include "word2vec.h"

#ifdef ZH
//using Model = Word2vec<std::u16string>;
#else
using Model = Word2vec<std::string>;
#endif
using Sentence = Model::Sentence;
using SentenceP = Model::SentenceP;


#ifdef ZH
const std::u16string MARKER = u"#m#";
std::vector<SentenceP> load_sentence(const std::string &path,bool with_marker,bool with_tag){
    auto is_word = [](char16_t ch){return ch >=0x4e00 && ch <= 0x9fff;}; //中日韩统一汉字 unihan
    auto close_tag = [](SentenceP& sentence){
        Model::Tag &t = sentence->tags_.back();
        if(t == Model::B) t = Model::S;
        else if(t == Model::M) t = Model::E;
    };
    size_t count = 0 ; 
    const size_t max_sentence_len = 200;
    std::vector<SentenceP> sentences;

    SentenceP sentence(new Sentence);
    std::ifstream in(path);
    while(1){
        std::string s;
        in >> s;

        if(s.empty()) break;
        std::u16string us = Cvt<std::u16string>::from_utf8(s);
        for(auto ch: us){
            if(is_word(ch)){
                if(sentence->tokens_.empty() && with_marker)
                    sentence->tokens_.push_back(MARKER);
                sentence->tokens_.push_back(std::u16string(1,ch));
                if(with_tag){
                    if(sentence->tags_.empty())
                        sentence->tags_.push_back(Model::B);
                    else{
                        auto &t = sentence->tags_.back();
                        Model::Tag nt = (t == Model::S||t==Model::E)?Model::B : Model::M;
                        sentence->tags_.push_back(nt);
                    }
                }
            }
            if(!is_word(ch) || sentence->tokens_.size() == max_sentence_len){ //不是句子开头的 不是unihan中的字符都用特殊字符MARKER表示
                if(sentence->tokens_.empty())continue;
                if(with_tag) close_tag(sentence);
                if(ch == u'，' || ch == '、')continue;
                if(with_marker) sentence->tokens_.push_back(MARKER);
                sentence->words_.reserve(sentence->tokens_.size());
                sentences.push_back(std::move(sentence));
                sentence.reset(new Sentence);
            }
        }
        if(!sentence->tokens_.empty() && with_tag) close_tag(sentence);
    }
    if(!sentence->tokens_.empty()){
        if(with_tag) close_tag(sentence);
        if(with_marker) sentence->tokens_.push_back(MARKER);
        sentences.push_back(std::move(sentence));
    }
    in.close();
    return sentences;
}

#else
std::vector<SentenceP> load_english(const std::string &path){
    std::ifstream in(path);
    std::vector<SentenceP> sentences;
    SentenceP sentence(new Sentence);
    const size_t max_sentence_len = 200;
    int count = 0;
    while(true){
        std::string s;
        in >> s;
        if(s.empty()) break;
        ++count;
        sentence->tokens_.push_back(std::move(s));
        if(count == max_sentence_len){
            count = 0;
            sentence->words_.reserve(sentence->tokens_.size());
            sentences.push_back(std::move(sentence));
            sentence.reset(new Sentence);
        }
    }
    if(!sentence->tokens_.empty())
        sentences.push_back(std::move(sentence));
    in.close();
    return sentences;
}
#endif
int accuracy(Model &model,std::string questions,int restrict_vocab=30000);
int main(int argc,char* argv[]){
    Model model;
    int n_workers = 4;
    ::srand(::time(NULL));
    auto distance = [&model](){
        while(1){
            std::string s;
            std::cout<<"\nFind nearest word for(:quit to break):";
            std::cin>>s;
            if(s == ":quit") break;
            std::vector<std::pair<std::string,float> > p = model.most_similar(std::vector<std::string>{s},std::vector<std::string>(),10);
            size_t i = 0;
            for(auto& v:p){
                std::cout<< i++ <<" " <<v.first<< " "<<v.second<<std::endl;
            }
        }
    };
    bool train = true;
    bool test = false;
    if(argc>1 && std::string(argv[1]) == "test"){
        std::swap(train,test);
    }
    if(train){
        std::string sfilename = "./text8";
        std::vector<SentenceP> sentences;// = load_english(sfilename);
        size_t count = 0;
        const size_t max_sentence_len = 200;
        SentenceP sentence(new Sentence);
        std::ifstream in("text8");
        while(true){
            std::string s;
            in >> s;
            if(s.empty()) break;
            ++count;
            sentence->tokens_.push_back(std::move(s));
            if(count == max_sentence_len){
                count = 0 ; 
                sentence->words_.reserve(sentence->tokens_.size());
                sentences.push_back(std::move(sentence));
                sentence.reset(new Sentence);
            }
        }
        if(!sentence->tokens_.empty()){
            sentences.push_back(std::move(sentence));
        }
        //std::cout<<sentences.size()<<" sentences, "<<std::accumulate(sentences.begin(),sentences.end(),(int)0,[](int x,const SentenceP &s){ return x+s->tokens_.size();})<<" words loaded."<<std::endl;

        auto cstart = std::chrono::high_resolution_clock::now();
        model.build_vocab(sentences);
        auto cend = std::chrono::high_resolution_clock::now();

        printf("load vocab: %.4f seconds\n",std::chrono::duration_cast<std::chrono::microseconds>(cend-cstart).count() / 1000000.0);
        cstart = cend;
        model.train(sentences,n_workers);
        cend = std::chrono::high_resolution_clock::now();
        printf("train: %.4f seconds\n",std::chrono::duration_cast<std::chrono::microseconds>(cend-cstart).count()/1000000.0);

        cstart = cend;
        model.save_text("./vectors.txt");
        cend = std::chrono::high_resolution_clock::now();
        printf("save_text: %.4f seconds\n",std::chrono::duration_cast<std::chrono::microseconds>(cend-cstart).count()/1000000.0);
    }
    if(test){
        std::cout<<"testing ..."<<std::endl;
        auto cstart = std::chrono::high_resolution_clock::now();
        model.load_text("vectors.txt");
        auto cend = std::chrono::high_resolution_clock::now();
        printf("load model: %.4f seconds\n",std::chrono::duration_cast<std::chrono::microseconds>(cstart-cend).count()/1000000.0);
        cstart = cend;
        accuracy(model,"questions-words.txt");
        cend = std::chrono::high_resolution_clock::now();
        printf("load model: %.4f seconds\n",std::chrono::duration_cast<std::chrono::microseconds>(cstart-cend).count()/1000000.0);

    }
    distance();
}

int accuracy(Model &model,std::string questions,int restrict_vocab){
    std::ifstream in(questions);
    std::string line;
    auto lower = [](std::string &data){
        std::transform(data.begin(),data.end(),data.begin(),::tolower);
    };
    size_t count = 0,correct = 0,ignore=0,almost_correct=0;
    const int topn = 10;
    while(std::getline(in,line)){
        if(line[0] == ':'){
            printf("%s\n",line.c_str());
            continue;
        }
        std::istringstream iss(line);
        std::string a,b,c,expected;
        iss>>a>>b>>c>>expected;
        lower(a);lower(b);lower(c);lower(expected);
        if(!model.has(a) || !model.has(b) || !model.has(c) || !model.has(expected)){
            printf("unhandled: %s %s %s %s\n",a.c_str(),b.c_str(),c.c_str(),expected.c_str());
            ++ignore;
            continue;
        }
        ++count;
        std::vector<std::string> positive{b,c},negtive{a};
        auto predict = model.most_similar(positive,negtive,topn);
        if(predict[0].first == expected){ ++correct; ++almost_correct;}
        else{
            bool found = false;
            for(auto &v : predict){
                if(v.first == expected){ found = true; break;}
            }
            if(found) ++almost_correct;
            else printf("predicted: %s, expected: %s\n",predict[0].first.c_str(),expected.c_str());
        }
    }
    if(count > 0)
        printf("predict %lu out of %lu (%f%%),almost correct %lu (%f%%) ignore %lu\n",correct,count,correct * 100.0 / count , almost_correct, almost_correct * 100.0 / count, ignore);
    return 0;
}

