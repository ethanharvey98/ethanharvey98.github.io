import { Component, OnInit } from '@angular/core';
import PublicationsJSON from '../../assets/publications.json';  

interface Publication {  
  year: Number;  
  reference: String;
  link?: String;
}

@Component({
  selector: 'app-publications',
  templateUrl: './publications.component.html',
  styleUrls: ['./publications.component.css']
})
export class PublicationsComponent implements OnInit {

  publications: Publication[] = PublicationsJSON;  
  years = new Set(this.publications.map(function(publication) {return publication.year;}));

  constructor() { }

  ngOnInit(): void {
  }

}
