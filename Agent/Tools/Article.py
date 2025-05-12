from pydantic import BaseModel

#Bu class aslında tek amacı güzel gözüktürmektir cevabimizi.Apinin cevabını(serpapi) daha toplu gösterir.
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str
    #Class olarak atama yapıp dönerbilgileri.Ayrıca bu pydantic classdaki otomatık __str__ sayesinde
    #classı print ederken direk düzgün olarak basıyor.
    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )