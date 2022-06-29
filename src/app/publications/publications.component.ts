import { Component, OnInit } from '@angular/core';
import PublicationsJSON from '../../assets/publications.json';  

interface Publication {  
  number: Number;  
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

  constructor() { }

  ngOnInit(): void {
  }

}
